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
import scipy.sparse
import torch
from tqdm.autonotebook import tqdm

from pykeen.datasets.base import EagerDataset
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.typing import MappedTriples
from pykeen.utils import compact_mapping

__all__ = [
    'Sealant',
    'get_candidate_inverse_relations',
    'unleak',
    'reindex',
]

logger = logging.getLogger(__name__)
X = TypeVar('X')
Y = TypeVar('Y')


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


def jaccard_similarity_scipy(
    a: scipy.sparse.spmatrix,
    b: scipy.sparse.spmatrix,
) -> numpy.ndarray:
    r"""Compute the Jaccard similarity between sets represented as sparse matrices.

    The similarity is computed as

    .. math ::
        J(A, B) = \frac{|A \cap B|}{|A \cup B|}
                = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}

    where the intersection can be computed in one batch as matrix product.

    :param a: shape: (m, max_num_elements)
        The first sets.
    :param b: shape: (n, max_num_elements)
        The second sets.

    :return: shape: (m, n)
        The pairwise Jaccard similarity.
    """
    sum_size = numpy.asarray(a.sum(axis=1) + b.sum(axis=1).T)
    intersection_size = numpy.asarray((a @ b.T).todense())
    # safe division for empty sets
    divisor = numpy.clip(sum_size - intersection_size, a_min=1, a_max=None)
    return intersection_size / divisor


def triples_factory_to_sparse_matrices(
    triples_factory: TriplesFactory,
) -> Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix]:
    """Compute relation representations as sparse matrices of entity pairs.

    .. note ::
        Both sets, head-tail-set, tail-head-set, have to be created at once since they need to share the same entity
        pair to Id mapping.

    :param triples_factory:
        The triples factory.

    :return: shape: (num_relations, num_entity_pairs)
        head-tail-set, tail-head-set matrices as {0, 1} integer matrices.
    """
    return mapped_triples_to_sparse_matrices(
        triples_factory.mapped_triples,
        num_relations=triples_factory.num_relations,
    )


def _to_one_hot(
    rows: torch.LongTensor,
    cols: torch.LongTensor,
    shape: Tuple[int, int],
) -> scipy.sparse.spmatrix:
    """Create a one-hot matrix given indices of non-zero elements (potentially containing duplicates)."""
    rows, cols = torch.stack([rows, cols], dim=0).unique(dim=1).numpy()
    values = numpy.ones(rows.shape[0], dtype=numpy.int32)
    return scipy.sparse.coo_matrix(
        (values, (rows, cols)),
        shape=shape,
        dtype=numpy.int32,
    )


def mapped_triples_to_sparse_matrices(
    mapped_triples: MappedTriples,
    num_relations: int,
) -> Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix]:
    """Compute relation representations as sparse matrices of entity pairs.

    .. note ::
        Both sets, head-tail-set, tail-head-set, have to be created at once since they need to share the same entity
        pair to Id mapping.

    :param mapped_triples:
        The input triples.
    :param num_relations:
        The number of input relations

    :return: shape: (num_relations, num_entity_pairs)
        head-tail-set, tail-head-set matrices as {0, 1} integer matrices.
    """
    num_triples = mapped_triples.shape[0]
    # compute unique pairs in triples *and* inverted triples for consistent pair-to-id mapping
    extended_mapped_triples = torch.cat(
        [
            mapped_triples,
            mapped_triples.flip(-1),
        ],
        dim=0,
    )
    pairs, pair_id = extended_mapped_triples[:, [0, 2]].unique(dim=0, return_inverse=True)
    n_pairs = pairs.shape[0]
    forward, backward = pair_id.split(num_triples)
    relations = mapped_triples[:, 1]
    rel = _to_one_hot(rows=relations, cols=forward, shape=(num_relations, n_pairs))
    inv = _to_one_hot(rows=relations, cols=backward, shape=(num_relations, n_pairs))
    return rel, inv


def get_candidate_pairs(
    *,
    a: scipy.sparse.spmatrix,
    b: Optional[scipy.sparse.spmatrix] = None,
    threshold: float,
    no_self: bool = True,
) -> Set[Tuple[int, int]]:
    """Find pairs of sets with Jaccard similarity above threshold using :func:`jaccard_similarity_scipy`.

    :param a:
        The first set.
    :param b:
        The second set. If not specified, reuse the first set.
    :param threshold:
        The threshold above which the similarity has to be.
    :param no_self:
        Whether to exclude (i, i) pairs.

    :return:
        A set of index pairs.
    """
    if b is None:
        b = a
    # duplicates
    sim = jaccard_similarity_scipy(a, b)
    if no_self:
        # we are not interested in self-similarity
        num = sim.shape[0]
        idx = numpy.arange(num)
        sim[idx, idx] = 0
    return set(zip(*(sim >= threshold).nonzero()))


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

        # compute similarities
        if symmetric:
            rel, inv = triples_factory_to_sparse_matrices(triples_factory=triples_factory)
            self.candidate_duplicate_relations = get_candidate_pairs(a=rel, threshold=self.minimum_frequency)
            self.candidate_inverse_relations = get_candidate_pairs(a=rel, b=inv, threshold=self.minimum_frequency)
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
        sizes = dict(zip(*triples_factory.mapped_triples[:, 1].unique(return_counts=True)))
        self.relations_to_delete = _select_by_most_pairs(
            components=_get_connected_components(pairs=((a, b) for (s, a, b) in self.candidates if (a != b))),
            size=sizes,
        )
        logger.info(f'identified {len(self.candidates)} from {self.triples_factory} to delete')

    def apply(self, triples_factory: TriplesFactory) -> TriplesFactory:
        """Make a new triples factory containing neither duplicate nor inverse relationships."""
        return triples_factory.new_with_restriction(relations=self.relations_to_delete, invert_relation_selection=True)


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
            (1, one_factory.relation_to_id),
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

    fb15k = get_dataset(dataset='fb15k')
    fb15k.summarize()

    n = 401  # magic 401 from the paper
    train, test, validate = unleak(fb15k.training, fb15k.testing, fb15k.validation, n=n)
    print()
    EagerDataset(train, test, validate).summarize(title='FB15k (cleaned)')

    fb15k237 = get_dataset(dataset='fb15k237')
    print('\nSummary FB15K-237')
    fb15k237.summarize()


if __name__ == '__main__':
    _main()
