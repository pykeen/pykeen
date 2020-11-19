# -*- coding: utf-8 -*-

"""Tools for removing the leakage from datasets.

Leakage is when the inverse of a given training triple appears in either
the testing or validation set. This scenario generally leads to inflated
and misleading evaluation because predicting an inverse triple is usually
very easy and not a sign of the generalizability of a model to predict
novel triples.
"""

import itertools as itt
import logging
from collections import Counter, defaultdict
from itertools import starmap
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple, TypeVar, Union

import numpy as np
from tabulate import tabulate
from tqdm.autonotebook import tqdm

from .triples_factory import TriplesFactory, create_entity_mapping, create_relation_mapping
from ..typing import LabeledTriples

__all__ = [
    'Sealant',
    'get_candidate_inverse_relations',
    'get_candidate_duplicate_relations',
    'unleak',
    'reindex',
    'summarize',
]

logger = logging.getLogger(__name__)
X = TypeVar('X')
Y = TypeVar('Y')


class Sealant:
    """Stores inverse frequencies and inverse mappings in a given triples factory."""

    triples_factory: TriplesFactory
    minimum_frequency: float
    inverses: Mapping[str, str]
    inverse_relations_to_delete: Set[str]

    def __init__(
        self,
        triples_factory: TriplesFactory,
        minimum_frequency: Optional[float] = None,
        symmetric: bool = True,
        use_tqdm: bool = True,
        use_multiprocessing: bool = False,
    ):
        """Index the inverse frequencies and the inverse relations in the triples factory.

        :param triples_factory: The triples factory to index.
        :param minimum_frequency: The minimum overlap between two relations' triples to consider them as inverses. The
         default value, 0.97, is taken from `Toutanova and Chen (2015) <https://www.aclweb.org/anthology/W15-4007/>`_,
         who originally described the generation of FB15k-237.
        """
        self.triples_factory = triples_factory
        if minimum_frequency is None:
            minimum_frequency = 0.97
        self.minimum_frequency = minimum_frequency

        if use_multiprocessing:
            use_tqdm = False

        self.candidate_duplicate_relations = get_candidate_duplicate_relations(
            triples_factory=self.triples_factory,
            minimum_frequency=self.minimum_frequency,
            symmetric=symmetric,
            use_tqdm=use_tqdm,
            use_multiprocessing=use_multiprocessing,
        )
        logger.info(
            f'identified {len(self.candidate_duplicate_relations)} candidate duplicate relationships'
            f' at similarity > {self.minimum_frequency} in {self.triples_factory}.',
        )
        self.duplicate_relations_to_delete = {r for r, _ in self.candidate_duplicate_relations}

        self.candidate_inverse_relations = get_candidate_inverse_relations(
            triples_factory=self.triples_factory,
            minimum_frequency=self.minimum_frequency,
            symmetric=symmetric,
            use_tqdm=use_tqdm,
            use_multiprocessing=use_multiprocessing,
        )
        logger.info(
            f'identified {len(self.candidate_inverse_relations)} candidate inverse pairs'
            f' at similarity > {self.minimum_frequency} in {self.triples_factory}',
        )

        if symmetric:
            self.inverses = dict(tuple(sorted(k)) for k in self.candidate_inverse_relations.keys())
            self.inverse_relations_to_delete = set(self.inverses.values())
        else:
            self.mutual_inverse = set()
            self.not_mutual_inverse = set()
            for r1, r2 in self.candidate_inverse_relations:
                if (r2, r1) in self.candidate_inverse_relations:
                    self.mutual_inverse.add((r1, r2))
                else:
                    self.not_mutual_inverse.add((r1, r2))
            logger.info(
                f'{len(self.mutual_inverse)} are mutual inverse ({len(self.mutual_inverse) // 2}'
                f' relations) and {len(self.not_mutual_inverse)} non-mutual inverse.',
            )

            # basically take all candidates
            self.inverses = dict(self.candidate_inverse_relations.keys())
            self.inverse_relations_to_delete = prioritize_mapping(self.candidate_inverse_relations)

        logger.info(f'identified {len(self.inverse_relations_to_delete)} from {self.triples_factory} to delete')

    @property
    def relations_to_delete(self) -> Set[str]:
        """Relations to delete combine from both duplicates and inverses."""
        return self.duplicate_relations_to_delete.union(self.inverse_relations_to_delete)

    def get_duplicate_triples(self, triples_factory: TriplesFactory) -> LabeledTriples:
        """Get labeled duplicate triples."""
        return triples_factory.get_triples_for_relations(self.duplicate_relations_to_delete)

    def new_without_duplicate_relations(self, triples_factory: TriplesFactory) -> TriplesFactory:
        """Make a new triples factory not containing duplicate relationships."""
        return triples_factory.new_without_relations(self.duplicate_relations_to_delete)

    def get_inverse_triples(self, triples_factory: TriplesFactory) -> LabeledTriples:
        """Get labeled inverse triples."""
        return triples_factory.get_triples_for_relations(self.inverse_relations_to_delete)

    def new_without_inverse_relations(self, triples_factory: TriplesFactory) -> TriplesFactory:
        """Make a new triples factory not containing inverse relationships."""
        return triples_factory.new_without_relations(self.inverse_relations_to_delete)

    def apply(self, triples_factory: TriplesFactory) -> TriplesFactory:
        """Make a new triples factory containing neither duplicate nor inverse relationships."""
        return triples_factory.new_without_relations(self.relations_to_delete)


def prioritize_mapping(d: Mapping[Tuple[str, str], float]) -> Set[str]:
    """Prioritize elements from a two way mapping."""
    return {
        b
        for a, b in d
        if (
            (b, a) not in d  # inverse didn't make the threshold
            or (d[a, b] == d[b, a] and a > b)  # inverse is equivalent, order by name
            or (d[a, b] < d[b, a])  # inverse isn't equivalent, use bigger similarity
        )
    }


def unleak(
    train: TriplesFactory,
    *triples_factories: TriplesFactory,
    n: Union[None, int, float] = None,
    minimum_frequency: Optional[float] = None,
) -> Iterable[TriplesFactory]:
    """Unleak a train, test, and validate triples factory.

    :param train: The target triples factory
    :param triples_factories: All other triples factories (test, validate, etc.)
    :param n: Either the (integer) number of top relations to keep or the (float) percentage of top relationships
     to keep. If left none, frequent relations are not removed.
    :param minimum_frequency: The minimum overlap between two relations' triples to consider them as inverses or
     duplicates. The default value, 0.97, is taken from
     `Toutanova and Chen (2015) <https://www.aclweb.org/anthology/W15-4007/>`_, who originally described the generation
     of FB15k-237.
    """
    if n is not None:
        frequent_relations = train.get_most_frequent_relations(n=n)
        logger.info(f'keeping most frequent relations from {train}')
        train = train.new_with_relations(frequent_relations)
        triples_factories = [
            triples_factory.new_with_relations(frequent_relations)
            for triples_factory in triples_factories
        ]

    # Calculate which relations are the inverse ones
    sealant = Sealant(train, minimum_frequency=minimum_frequency)

    if not sealant.relations_to_delete:
        logger.info(f'no relations to delete identified from {train}')
    else:
        train = sealant.apply(train)
        triples_factories = [
            sealant.apply(triples_factory)
            for triples_factory in triples_factories
        ]

    return reindex(train, *triples_factories)


def reindex(*triples_factories: TriplesFactory) -> List[TriplesFactory]:
    """Reindex a set of triples factories."""
    triples = np.concatenate(
        [
            triples_factory.triples
            for triples_factory in triples_factories
        ],
        axis=0,
    )
    entity_to_id = create_entity_mapping(triples)
    relation_to_id = create_relation_mapping(set(triples[:, 1]))

    return [
        TriplesFactory(
            triples=triples_factory.triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            # FIXME doesn't carry flag of create_inverse_triples through
        )
        for triples_factory in triples_factories
    ]


def summarize(training, testing, validation) -> None:
    """Summarize the dataset."""
    headers = ['Set', 'Entities', 'Relations', 'Triples']
    print(tabulate(
        [
            ['Train', training.num_entities, training.num_relations, training.num_triples],
            ['Test', testing.num_entities, testing.num_relations, testing.num_triples],
            ['Valid', validation.num_entities, validation.num_relations, validation.num_triples],
        ],
        headers=headers,
    ))


def get_candidate_inverse_relations(
    triples_factory: TriplesFactory,
    *,
    symmetric: bool = True,
    minimum_frequency: Optional[float] = None,
    skip_zeros: bool = True,
    skip_self: bool = True,
    use_tqdm: bool = True,
    use_multiprocessing=False,
) -> Mapping[Tuple[str, str], float]:
    """Count which relationships might be inverses of each other.

    :param symmetric: Should set similarity be calculated as the Jaccard index (symmetric) or as the
     set inclusion percentage (asymmetric)?
    :param minimum_frequency: If set, pairs of relations and candidate inverse relations
     with a similarity lower than this value will not be reported.
    :param skip_zeros: Should similarities between forward and candidate inverses
     of `0.0` be discarded?
    :param skip_self: Should similarities between a relationship and its own
     candidate inverse be skipped? Defaults to True, but could be useful to identify
     relationships that aren't directed.
    :param use_tqdm: Should :mod:`tqdm` be used to track progress of the similarity calculations?
    :param use_multiprocessing: Should :mod:`multiprocessing` be used to offload the similarity calculations across
     multiple cores?
    :return: A counter whose keys are pairs of relations and values are similarity scores
    """
    # A dictionary of all of the head/tail pairs for a given relation
    relations: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    # A dictionary for all of the tail/head pairs for a given relation
    candidate_inverse_relations: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for h, r, t in triples_factory.triples:
        relations[r].add((h, t))
        candidate_inverse_relations[r].add((t, h))

    # Calculate the similarity between each relationship (entries in ``forward``)
    # with all other candidate inverse relationships (entries in ``inverse``)
    if symmetric:
        it = (
            ((r1, relations[r1]), (r2, candidate_inverse_relations[r2]))
            for r1, r2 in itt.combinations(relations, 2)
        )
        total = int(len(relations) * (len(relations) - 1) // 2)
    else:
        # Note: uses an asymmetric metric, so results for ``(a, b)`` is not necessarily the
        # same as for ``(b, a)``
        it = itt.product(relations.items(), candidate_inverse_relations.items())
        total = int(len(relations) ** 2)

    if use_tqdm:
        it = tqdm(it, total=total, desc='getting candidate inverse relations')
    return _check_similar_sets(
        it,
        skip_zeros=skip_zeros,
        skip_self=skip_self,
        minimum_frequency=minimum_frequency,
        symmetric=symmetric,
        use_multiprocessing=use_multiprocessing,
    )


def get_candidate_duplicate_relations(
    triples_factory: TriplesFactory,
    *,
    minimum_frequency: Optional[float] = None,
    skip_zeros: bool = True,
    symmetric: bool = True,
    use_tqdm: bool = True,
    use_multiprocessing: bool = False,
):
    """Count which relationships might be duplicates.

    :param symmetric: Should set similarity be calculated as the Jaccard index (symmetric) or as the
     set inclusion percentage (asymmetric)?
    :param minimum_frequency: If set, pairs of relations and candidate inverse relations
     with a similarity lower than this value will not be reported.
    :param skip_zeros: Should similarities between forward and candidate inverses
     of `0.0` be discarded?
    :param use_tqdm: Should :mod:`tqdm` be used to track progress of the similarity calculations?
    :param use_multiprocessing: Should :mod:`multiprocessing` be used to offload the similarity calculations across
     multiple cores?
    :return: A counter whose keys are pairs of relations and values are similarity scores
    """
    # A dictionary of all of the head/tail pairs for a given relation
    relations: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for h, r, t in triples_factory.triples:
        relations[r].add((h, t))

    it = itt.combinations(relations.items(), 2)
    if use_tqdm:
        it = tqdm(it, total=len(relations) * (len(relations) - 1) / 2, desc='getting candidate duplicate relations')
    return _check_similar_sets(
        it,
        skip_zeros=skip_zeros,
        skip_self=False,
        minimum_frequency=minimum_frequency,
        symmetric=symmetric,
        use_multiprocessing=use_multiprocessing,
    )


def _check_similar_sets(
    it: Iterable[Tuple[Tuple[X, Y], Tuple[X, Y]]],
    *,
    skip_zeros: bool,
    skip_self: bool,
    minimum_frequency: Optional[float] = None,
    symmetric: bool = True,
    use_multiprocessing: bool = True,
) -> Mapping[Tuple[X, X], float]:
    if symmetric:
        _similarity_metric = _get_jaccard_index_unwrapped
    else:
        _similarity_metric = _get_asymmetric_jaccard_index_unwrapped

    if not skip_self:
        rv = (
            (r1, r1_pairs, r2, r2_pairs)
            for (r1, r1_pairs), (r2, r2_pairs) in it
        )
    else:
        # Filter out results between a given relationship and itself
        rv = (
            (r1, r1_pairs, r2, r2_pairs)
            for (r1, r1_pairs), (r2, r2_pairs) in it
            if r1 != r2
        )

    if use_multiprocessing:
        logger.info('using multiprocessing')
        with Pool(cpu_count()) as pool:
            rv = pool.starmap(_similarity_metric, rv)
    else:
        rv = starmap(_similarity_metric, rv)

    if skip_zeros and minimum_frequency is None:
        minimum_frequency = 0.0

    if minimum_frequency is not None:
        # Filter out results below a minimum frequency
        rv = (
            ((r1, r2), similarity)
            for (r1, r2), similarity in rv
            if minimum_frequency < similarity
        )

    return Counter(dict(rv))


def _get_asymmetric_jaccard_index(a: Set[X], b: Set[X]) -> float:
    if a:
        return len(a.intersection(b)) / len(a)
    return 0.0


def _get_jaccard_index(a: Set[X], b: Set[X]) -> float:
    if a and b:
        return len(a.intersection(b)) / len(a.union(b))
    return 0.0


def _get_szymkiewicz_simpson_coefficient(a: Set[X], b: Set[X]) -> float:
    """Calculate the Szymkiewicz–Simpson coefficient.

    .. seealso:: https://en.wikipedia.org/wiki/Overlap_coefficient
    """
    if a and b:
        return len(a.intersection(b)) / min(len(a), len(b))
    return 0.0


def _get_jaccard_index_unwrapped(r1: X, r1_pairs, r2, r2_pairs) -> Tuple[Tuple[X, X], float]:
    return (r1, r2), _get_jaccard_index(r1_pairs, r2_pairs)


def _get_asymmetric_jaccard_index_unwrapped(
    r1: X, r1_pairs: Set[Y], r2: X, r2_pairs: Set[Y],
) -> Tuple[Tuple[X, X], float]:
    return (r1, r2), _get_asymmetric_jaccard_index(r1_pairs, r2_pairs)


def _main():
    """Test unleaking FB15K.

    Run with ``python -m pykeen.triples.leakage``.
    """
    from pykeen.datasets import get_dataset
    logging.basicConfig(format='pykeen: %(message)s', level=logging.INFO)

    print('Summary FB15K')
    fb15k = get_dataset(dataset='fb15k')
    summarize(fb15k.training, fb15k.testing, fb15k.validation)

    print('\nSummary FB15K (cleaned)')
    n = 401  # magic 401 from the paper
    train, test, validate = unleak(fb15k.training, fb15k.testing, fb15k.validation, n=n)
    summarize(train, test, validate)

    print('\nSummary FB15K-237')
    fb15k237 = get_dataset(dataset='fb15k237')
    summarize(fb15k237.training, fb15k237.testing, fb15k237.validation)


if __name__ == '__main__':
    _main()
