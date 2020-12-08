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
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Set, Tuple, TypeVar, Union

import numpy
import torch
from tabulate import tabulate
from tqdm.autonotebook import tqdm

from pykeen.triples.triples_factory import TriplesFactory
from pykeen.typing import MappedTriples
from pykeen.utils import compact_mapping

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


def _jaccard_similarity_join(
    sets: Mapping[int, Set[Tuple[X, X]]],
    inverse_sets: Mapping[int, Set[Tuple[X, X]]],
    threshold: float = 0.0,
) -> Tuple[Collection[Tuple[float, X, X]], Collection[Tuple[float, X, X]]]:
    r"""
    Compute Jaccard similarity between pairs of sets.

    .. math ::
        J(A, B) = \frac{|A \cap B|}{|A \cup B|}

    The method returns the true value for all pairs with similarity larger than the threshold. For pairs which are
    guaranteed to be less similar, it may skip to compute the exact value to accelerate. For this, it makes use of
    the following inequality:

    .. math ::
        J(A, B) = \frac{|A \cap B|}{|A \cup B|}
                \leq \frac{\min (|A|, |B|)}{\max (|A|, |B|)}

    Hence, if :math:`\frac{\min (|A|, |B|)}{\max (|A|, |B|)} < \tau`, we know for sure that :math:`J(A, B) < \tau`.

    :param sets:
        The sets of tuples.
    :param inverse_sets:
        The sets of inverted tuples.

    :return:
        A similarity matrix. The lower triangular matrix contains the Jaccard similarity between the sets.
        The upper diagonal the similarity to the sets with inverted tuples. The diagonal contains teh similarity
        between the set and the same set with inverted tuples.
    """
    keys = sorted(sets.keys())
    n_relations = len(keys)
    assert set(sets.keys()) == set(inverse_sets.keys())

    # The individual sizes (note that len(inv_set) = len(set)
    size = numpy.asarray([len(sets[r]) for r in keys])
    ub = numpy.minimum(size[None, :], size[:, None]) / numpy.maximum(size[None, :], size[:, None])
    ub[numpy.arange(n_relations), numpy.arange(n_relations)] = 0
    ub = numpy.triu(ub)
    candidates = list(zip(*(ub > threshold).nonzero()))
    max_n_candidates = n_relations * (n_relations - 1) // 2
    logger.info(
        f"Reduced candidates from {max_n_candidates} to {len(candidates)} "
        f"(reduction by {1 - len(candidates) / max_n_candidates:2.2%}) using upper bound."
    )

    # compute Jaccard similarity:
    # J = |A n B| / |A u B|
    # Thus, J = 1 / J' with J' = |A u B| / |A n B| = (|A| + |B| + |A n B|) / |A n B| = (|A| + |B|)/(|A n B|) - 1
    # we are not interested in self-similarity, thus we set it to zero
    duplicates = []
    inverses = []
    for i, j in tqdm(candidates, unit="pair", unit_scale=True):
        assert j >= i
        ri, rj = [keys[k] for k in (i, j)]
        pi, pj, pji = [sets[r] for r in (ri, rj)] + [inverse_sets[rj]]
        # J(P_i, P_j) > tau <=> 1 / J' > tau <=> |P_i n P_j| > tau * (|P_i| + |P_j|)
        size_sum = (size[i] + size[j])
        i_ij = len(pi.intersection(pj))
        if i_ij > threshold * size_sum:
            duplicates.append((1.0 / (size_sum / i_ij - 1), ri, rj))
        i_ij_i = len(pi.intersection(pji))
        if i_ij_i > threshold * size_sum:
            inverses.append((1.0 / (size_sum / i_ij_i - 1), ri, rj))
    # symmetric similarity: add both pairs
    duplicates.extend((a, s, r) for a, r, s in duplicates)
    inverses.extend((a, s, r) for a, r, s in inverses)
    return duplicates, inverses


def find(x: X, parent: Mapping[X, X]) -> X:
    # check validity
    if x not in parent:
        raise ValueError(f'Unknown element: {x}.')
    # path compression
    while parent[x] != x:
        x, parent[x] = parent[x], parent[parent[x]]
    return x


def _get_connected_components(pairs: Iterable[Tuple[X, X]]) -> Collection[Collection[X]]:
    # collect connected components using union find with path compression
    parent = dict()
    for x, y in pairs:
        parent.setdefault(x, x)
        parent.setdefault(y, y)
        # get representatives
        x = find(x=x, parent=parent)
        y = find(x=y, parent=parent)
        # already merged
        if x == y:
            continue
        # make x the smaller one
        if y < x:
            x, y = y, x
        # merge
        parent[y] = x
    # extract partitions
    result = defaultdict(list)
    for k, v in parent.items():
        result[v].append(k)
    return list(result.values())


def _select_by_most_pairs(
    components: Collection[Collection[int]],
    size: Mapping[int, int],
) -> Collection[int]:
    """Select relations to keep with the most associated pairs."""
    result = set()
    for component in components:
        keep = max(component, key=size.__getitem__)
        result.update(r for r in component if r != keep)
    return result


class Sealant:
    """Stores inverse frequencies and inverse mappings in a given triples factory."""

    triples_factory: TriplesFactory
    minimum_frequency: float
    inverses: Mapping[int, int]
    inverse_relations_to_delete: Set[int]

    def __init__(
        self,
        triples_factory: TriplesFactory,
        minimum_frequency: Optional[float] = None,
        symmetric: bool = True,
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

        # convert relations to sparse adjacency matrices
        relations = defaultdict(set)
        inv_relations = defaultdict(set)
        for (h, r, t) in triples_factory.mapped_triples.tolist():
            relations[r].add((h, t))
            inv_relations[r].add((t, h))

        if symmetric:
            self.candidate_duplicate_relations, self.candidate_inverse_relations = _jaccard_similarity_join(
                sets=relations,
                inverse_sets=inv_relations,
                threshold=self.minimum_frequency,
            )
        else:
            raise NotImplementedError
        logger.info(
            f'identified {len(self.candidate_duplicate_relations)} candidate duplicate relationships'
            f' at similarity > {self.minimum_frequency} in {self.triples_factory}.',
        )
        logger.info(
            f'identified {len(self.candidate_inverse_relations)} candidate inverse pairs'
            f' at similarity > {self.minimum_frequency} in {self.triples_factory}',
        )
        self.candidates = set(self.candidate_duplicate_relations).union(self.candidate_inverse_relations)
        self.relations_to_delete = _select_by_most_pairs(
            components=_get_connected_components(pairs=((a, b) for (s, a, b) in self.candidates)),
            size={r: len(pairs) for r, pairs in relations.items()},
        )
        logger.info(f'identified {len(self.candidates)} from {self.triples_factory} to delete')

    def apply(self, triples_factory: TriplesFactory) -> TriplesFactory:
        """Make a new triples factory containing neither duplicate nor inverse relationships."""
        return triples_factory.new_with_restriction(relations=self.relations_to_delete, invert_relation_selection=True)


def prioritize_mapping(d: Mapping[Tuple[X, X], float]) -> Set[X]:
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
        train = train.new_with_restriction(relations=frequent_relations)
        triples_factories = [
            triples_factory.new_with_restriction(relations=frequent_relations)
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


def _generate_compact_vectorized_lookup(
    ids: torch.LongTensor,
    label_to_id: Mapping[str, int],
) -> Tuple[Mapping[str, int], torch.LongTensor]:
    """
    Given a tensor of IDs and a label to ID mapping, retain only occurring IDs, and compact the mapping.

    Additionally returns a vectorized translation, i.e. a tensor `translation` of shape (max_old_id,) with
    `translation[old_id] = new_id` for all translated IDs and `translation[old_id] = -1` for non-occurring IDs.
    This allows to use `translation[ids]` to translate the IDs as a simple integer index based lookup.

    :param ids:
        The tensor of IDs.
    :param label_to_id:
        The label to ID mapping.

    :return:
        A tuple new_label_to_id, vectorized_translation.
    """
    # get existing IDs
    existing_ids = set(ids.view(-1).unique().tolist())
    # remove non-existing ID from label mapping
    label_to_id, old_to_new_id = compact_mapping(mapping={
        label: i
        for label, i in label_to_id.items()
        if i in existing_ids
    })
    # create translation tensor
    translation = torch.full(size=(max(existing_ids) + 1,), fill_value=-1)
    for old, new in old_to_new_id.items():
        translation[old] = new
    return label_to_id, translation


def _translate_triples(
    triples: MappedTriples,
    entity_translation: torch.LongTensor,
    relation_translation: torch.LongTensor,
) -> MappedTriples:
    """
    Translate triples given vectorized translations for entities and relations.

    :param triples: shape: (num_triples, 3)
        The original triples
    :param entity_translation: shape: (num_old_entity_ids,)
        The translation from old to new entity IDs.
    :param relation_translation: shape: (num_old_relation_ids,)
        The translation from old to new relation IDs.

    :return: shape: (num_triples, 3)
        The translated triples.
    """
    triples = torch.stack(
        [
            trans[column]
            for column, trans in zip(
            triples.t(),
            (entity_translation, relation_translation, entity_translation),
        )
        ],
        dim=-1,
    )
    assert (triples >= 0).all()
    return triples


def reindex(*triples_factories: TriplesFactory) -> List[TriplesFactory]:
    """Reindex a set of triples factories."""
    # get entities and relations occurring in triples
    all_triples = torch.cat([
        factory.mapped_triples
        for factory in triples_factories
    ], dim=0)

    # generate ID translation and new label to Id mappings
    one_factory = triples_factories[0]
    (entity_to_id, entity_id_translation), (relation_to_id, relation_id_translation) = [
        _generate_compact_vectorized_lookup(
            ids=all_triples[:, cols],
            label_to_id=label_to_id,
        )
        for cols, label_to_id in (
            ([0, 2], one_factory.entity_to_id),
            (1, one_factory.relation_to_id)
        )
    ]

    return [
        TriplesFactory(
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            mapped_triples=_translate_triples(
                triples=factory.mapped_triples,
                entity_translation=entity_id_translation,
                relation_translation=relation_id_translation,
            ),
            create_inverse_triples=factory.create_inverse_triples,
        )
        for factory in triples_factories
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
) -> Mapping[Tuple[int, int], float]:
    """Count which relationships might be inverses of each other.

    :param triples_factory:
        The triples factory.
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
    # TODO: Deprecated
    # A dictionary of all of the head/tail pairs for a given relation
    relations: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    # A dictionary for all of the tail/head pairs for a given relation
    candidate_inverse_relations: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for h, r, t in triples_factory.mapped_triples.tolist():
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
) -> Mapping[Tuple[int, int], float]:
    """Count which relationships might be duplicates.

    :param triples_factory:
        The triples factory.
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
    # TODO: Deprecated
    # A dictionary of all of the head/tail pairs for a given relation
    relations: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for h, r, t in triples_factory.mapped_triples.tolist():
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
    # TODO: Deprecated
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
