"""Analysis utilities for (mapped) triples."""

import hashlib
import logging
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse
from tqdm.auto import tqdm

from . import TriplesFactory
from ..constants import COLUMN_LABELS, TARGET_TO_INDEX
from ..typing import COLUMN_HEAD, COLUMN_RELATION, COLUMN_TAIL, LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, MappedTriples

logger = logging.getLogger(__name__)

__all__ = [
    "add_entity_labels",
    "add_relation_labels",
    "entity_relation_co_occurrence",
    "get_entity_counts",
    "get_relation_counts",
    "get_relation_functionality",
    "relation_cardinality_types",
    "relation_injectivity",
    "relation_pattern_types",
]

# constants
CARDINALITY_TYPE_ONE_TO_ONE = "one-to-one"
CARDINALITY_TYPE_ONE_TO_MANY = "one-to-many"
CARDINALITY_TYPE_MANY_TO_ONE = "many-to-one"
CARDINALITY_TYPE_MANY_TO_MANY = "many-to-many"

RELATION_CARDINALITY_TYPES = {
    CARDINALITY_TYPE_ONE_TO_ONE,
    CARDINALITY_TYPE_ONE_TO_MANY,
    CARDINALITY_TYPE_MANY_TO_ONE,
    CARDINALITY_TYPE_MANY_TO_MANY,
}

# constants
PATTERN_TYPE_SYMMETRY = "symmetry"
PATTERN_TYPE_ANTI_SYMMETRY = "anti-symmetry"
PATTERN_TYPE_INVERSION = "inversion"
PATTERN_TYPE_COMPOSITION = "composition"

RELATION_PATTERN_TYPES = {
    # unary
    PATTERN_TYPE_SYMMETRY,
    PATTERN_TYPE_ANTI_SYMMETRY,
    # binary
    PATTERN_TYPE_INVERSION,
    # ternary
    PATTERN_TYPE_COMPOSITION,
}

# column names
COUNT_COLUMN_NAME = "count"
ENTITY_ID_COLUMN_NAME = "entity_id"
RELATION_ID_COLUMN_NAME = "relation_id"
ENTITY_POSITION_COLUMN_NAME = "entity_position"
RELATION_LABEL_COLUMN_NAME = "relation_label"
ENTITY_LABEL_COLUMN_NAME = "entity_label"
INVERSE_FUNCTIONALITY_COLUMN_NAME = "inverse_functionality"
FUNCTIONALITY_COLUMN_NAME = "functionality"
CARDINALITY_TYPE_COLUMN_NAME = "relation_type"
PATTERN_TYPE_COLUMN_NAME = "pattern"
CONFIDENCE_COLUMN_NAME = "confidence"
SUPPORT_COLUMN_NAME = "support"


def _add_labels(
    df: pd.DataFrame,
    add_labels: bool,
    label_to_id: Mapping[str, int] | None,
    id_column: str,
    label_column: str,
    label_to_id_mapping_name: str,
    triples_factory: TriplesFactory | None = None,
) -> pd.DataFrame:
    """Add labels to a dataframe."""
    if not add_labels:
        return df
    if not label_to_id:
        if not triples_factory:
            raise ValueError
        label_to_id = getattr(triples_factory, label_to_id_mapping_name)
    if label_to_id is None:
        raise ValueError("label_to_id must not be None.")
    return df.merge(
        right=pd.DataFrame(
            data=list(label_to_id.items()),
            columns=[label_column, id_column],
        ),
        on=id_column,
    )


def add_entity_labels(
    *,
    df: pd.DataFrame,
    add_labels: bool,
    label_to_id: Mapping[str, int] | None = None,
    triples_factory: TriplesFactory | None = None,
) -> pd.DataFrame:
    """Add entity labels to a dataframe."""
    return _add_labels(
        df=df,
        add_labels=add_labels,
        label_to_id=label_to_id,
        id_column=ENTITY_ID_COLUMN_NAME,
        label_column=ENTITY_LABEL_COLUMN_NAME,
        label_to_id_mapping_name="entity_to_id",
        triples_factory=triples_factory,
    )


def add_relation_labels(
    df: pd.DataFrame,
    *,
    add_labels: bool,
    label_to_id: Mapping[str, int] | None = None,
    triples_factory: TriplesFactory | None = None,
) -> pd.DataFrame:
    """Add relation labels to a dataframe."""
    return _add_labels(
        df=df,
        add_labels=add_labels,
        label_to_id=label_to_id,
        id_column=RELATION_ID_COLUMN_NAME,
        label_column=RELATION_LABEL_COLUMN_NAME,
        label_to_id_mapping_name="relation_to_id",
        triples_factory=triples_factory,
    )


class PatternMatch(NamedTuple):
    """A pattern match tuple of relation_id, pattern_type, support, and confidence."""

    relation_id: int
    pattern_type: str
    support: int
    confidence: float


class _PairIndex(NamedTuple):
    """An index of the distinct triples, and of their distinct (head, tail) pairs."""

    #: the distinct triples' heads, relations, and tails, shape: (num_triples,)
    heads: np.ndarray
    relations: np.ndarray
    tails: np.ndarray

    #: the occurring relation IDs in the order of their first occurrence
    relation_order: np.ndarray

    #: an upper bound on entity / relation IDs
    num_entities: int
    num_relations: int

    #: the sorted keys `head * num_entities + tail` of distinct pairs, shape: (num_pairs,)
    pair_keys: np.ndarray

    #: a binary incidence matrix of pairs and relations, shape: (num_pairs, num_relations)
    incidence: scipy.sparse.csr_array

    def lookup_pairs(self, heads: np.ndarray, tails: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Look up pair IDs for (head, tail) pairs, returning the IDs of existing pairs, and a mask of existence."""
        keys = heads * self.num_entities + tails
        ids = np.searchsorted(self.pair_keys, keys)
        mask = ids < len(self.pair_keys)
        mask[mask] = self.pair_keys[ids[mask]] == keys[mask]
        return ids[mask], mask


def _build_pair_index(mapped_triples: Collection[tuple[int, int, int]]) -> _PairIndex:
    """Build a pair index from ID-based triples."""
    triples = np.asarray(mapped_triples, dtype=np.int64).reshape(-1, 3)
    # relations in the order of first occurrence
    relation_ids, first_index = np.unique(triples[:, 1], return_index=True)
    relation_order = relation_ids[np.argsort(first_index)]
    num_entities = int(triples[:, [0, 2]].max(initial=-1)) + 1
    num_relations = int(triples[:, 1].max(initial=-1)) + 1
    if num_entities**2 > np.iinfo(np.int64).max:
        raise ValueError(f"Too many entities for an int64-based pair index: {num_entities}")
    # distinct triples
    heads, relations, tails = np.unique(triples, axis=0).T
    # distinct pairs
    pair_keys, pair_ids = np.unique(heads * num_entities + tails, return_inverse=True)
    incidence = scipy.sparse.csr_array(
        (np.ones_like(pair_ids), (pair_ids, relations)), shape=(len(pair_keys), num_relations)
    )
    return _PairIndex(
        heads=heads,
        relations=relations,
        tails=tails,
        relation_order=relation_order,
        num_entities=num_entities,
        num_relations=num_relations,
        pair_keys=pair_keys,
        incidence=incidence,
    )


class _PairStatistics(NamedTuple):
    """Pair-overlap statistics for relations, in order of their first occurrence."""

    #: the relation IDs, shape: (n,)
    relations: np.ndarray

    #: overlap[i, j] = |{(x, y) : r_i(x, y) and r_j(x, y)}|, shape: (n, n)
    #: in particular, the diagonal contains the number of distinct pairs per relation
    overlap: np.ndarray

    #: reverse_overlap[i, j] = |{(x, y) : r_i(x, y) and r_j(y, x)}|, shape: (n, n)
    reverse_overlap: np.ndarray


def _pair_statistics(index: _PairIndex) -> _PairStatistics:
    """Compute pair-overlap statistics via sparse matrix products."""
    m = index.incidence
    # the rows of m_rev are the rows of m for the reversed pair, i.e., m_rev[(x, y), r] = r(y, x)
    reverse_ids, mask = index.lookup_pairs(heads=index.tails, tails=index.heads)
    m_rev = scipy.sparse.csr_array((np.ones_like(reverse_ids), (reverse_ids, index.relations[mask])), shape=m.shape)
    m_t = m.transpose().tocsr()
    rs = index.relation_order
    return _PairStatistics(
        relations=rs,
        overlap=(m_t @ m).toarray()[np.ix_(rs, rs)],
        reverse_overlap=(m_t @ m_rev).toarray()[np.ix_(rs, rs)],
    )


def _iter_pattern_matches(
    pattern_type: str,
    relation_ids: np.ndarray,
    support: np.ndarray,
    confidence: np.ndarray,
    skip_zero: bool = False,
) -> Iterable[PatternMatch]:
    """Convert aligned arrays to pattern match tuples (of Python scalars), optionally skipping zero confidence."""
    if skip_zero:
        mask = confidence > 0
        relation_ids, support, confidence = relation_ids[mask], support[mask], confidence[mask]
    for relation_id, supp, conf in zip(relation_ids.tolist(), support.tolist(), confidence.tolist(), strict=True):
        yield PatternMatch(relation_id, pattern_type, supp, conf)


def _iter_unary_patterns(stats: _PairStatistics, skip_zero: bool = False) -> Iterable[PatternMatch]:
    """Yield unary patterns from pair statistics, cf. :func:`iter_unary_patterns`."""
    logger.debug("Evaluating unary patterns: {symmetry, anti-symmetry}")
    support = np.diag(stats.overlap)
    confidence = np.diag(stats.reverse_overlap) / support
    yield from _iter_pattern_matches(PATTERN_TYPE_SYMMETRY, stats.relations, support, confidence, skip_zero)
    # confidence(anti-symmetry) = |ht \ rev(ht)| / |ht| = 1 - confidence(symmetry)
    yield from _iter_pattern_matches(PATTERN_TYPE_ANTI_SYMMETRY, stats.relations, support, 1 - confidence, skip_zero)


def _iter_binary_patterns(stats: _PairStatistics, skip_zero: bool = False) -> Iterable[PatternMatch]:
    """Yield binary patterns from pair statistics, cf. :func:`iter_binary_patterns`."""
    logger.debug("Evaluating binary patterns: {inversion}")
    # only pairs (r', r) where r' occurs (first) before r
    i, j = np.triu_indices(len(stats.relations), k=1)
    support = np.diag(stats.overlap)[i]
    # note: this uses `overlap` (r'(x, y) => r(x, y)) rather than `reverse_overlap` (r'(x, y) => r(y, x))
    confidence = stats.overlap[i, j] / support
    yield from _iter_pattern_matches(PATTERN_TYPE_INVERSION, stats.relations[j], support, confidence, skip_zero)


def iter_unary_patterns(
    mapped_triples: Collection[tuple[int, int, int]],
) -> Iterable[PatternMatch]:
    r"""
    Yield unary patterns from ID-based triples.

    =============  ===============================
    Pattern        Equation
    =============  ===============================
    Symmetry       $r(x, y) \implies r(y, x)$
    Anti-Symmetry  $r(x, y) \implies \neg r(y, x)$
    =============  ===============================

    .. note ::
        By definition, we have confidence(anti-symmetry) = 1 - confidence(symmetry).

    :param mapped_triples:
        A collection of ID-based triples.

    :yields: A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    yield from _iter_unary_patterns(_pair_statistics(_build_pair_index(mapped_triples)))


def iter_binary_patterns(
    mapped_triples: Collection[tuple[int, int, int]],
) -> Iterable[PatternMatch]:
    r"""
    Yield binary patterns from ID-based triples.

    =========  ===========================
    Pattern    Equation
    =========  ===========================
    Inversion  $r'(x, y) \implies r(y, x)$
    =========  ===========================

    Relation pairs $(r', r)$ are only considered for $r'$ occurring before $r$ in the triples.

    :param mapped_triples:
        A collection of ID-based triples.

    :yields: A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    yield from _iter_binary_patterns(_pair_statistics(_build_pair_index(mapped_triples)))


class _CompositionStatistics(NamedTuple):
    r"""Statistics for composition patterns $r_1(x, y) \land r_2(y, z) \implies r(x, z)$."""

    #: the relation pairs (r_1, r_2) with non-empty support, shape: (num_candidates,)
    first: np.ndarray
    second: np.ndarray

    #: the number of distinct (x, z) with r_1(x, y) and r_2(y, z) for some y, shape: (num_candidates,)
    support: np.ndarray

    #: hits[c, r] = number of such (x, z) with r(x, z), shape: (num_candidates, num_relations)
    hits: scipy.sparse.csr_array


def _composition_statistics(index: _PairIndex, chunk_size: int = 2**24) -> _CompositionStatistics:
    """Compute composition statistics via a chunked sparse self-join.

    :param index:
        The pair index.
    :param chunk_size:
        The (approximate) maximum number of joined rows to materialize at once. The join is chunked over groups of
        $(r_1, x)$, such that a joined $(r_1, x, r_2, z)$ never spans multiple chunks. A single group exceeding this
        size is processed as a chunk of its own.

    :return:
        The composition statistics.
    """
    num_e, num_r = index.num_entities, index.num_relations
    # right-hand side atoms r_2(y, z) as a binary matrix of shape (num_entities, num_relations * num_entities)
    right = scipy.sparse.csr_array(
        (np.ones_like(index.heads), (index.heads, index.relations * num_e + index.tails)),
        shape=(num_e, num_r * num_e),
    )
    # left-hand side atoms r_1(x, y) as a binary matrix of shape (num_groups, num_entities), one row per (r_1, x)
    group_keys, group_ids = np.unique(index.relations * num_e + index.heads, return_inverse=True)
    left = scipy.sparse.csr_array((np.ones_like(group_ids), (group_ids, index.tails)), shape=(len(group_keys), num_e))
    group_r, group_x = np.divmod(group_keys, num_e)
    # chunk over groups by the size of the (non-deduplicated) join
    out_degree = np.bincount(index.heads, minlength=num_e)
    join_size = np.bincount(group_ids, weights=out_degree[index.tails], minlength=len(group_keys)).astype(np.int64)
    chunk_ids = (np.cumsum(join_size) - join_size) // chunk_size
    bounds = np.r_[0, np.flatnonzero(chunk_ids[1:] != chunk_ids[:-1]) + 1, len(group_keys)]

    support_keys: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    support_counts: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    hit_keys: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    hit_relations: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    hit_counts: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    for start, stop in tqdm(
        zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True),
        total=len(bounds) - 1,
        desc="Checking ternary patterns",
        unit="chunk",
        unit_scale=True,
    ):
        # self-join r_1(x, y) & r_2(y, z) on y; the sparse product yields each distinct (r_1, x, r_2, z) once
        joined = (left[start:stop] @ right).tocoo()
        rows = start + joined.row
        r1, x = group_r[rows], group_x[rows]
        r2, z = np.divmod(joined.col.astype(np.int64), num_e)
        # support: number of distinct (x, z) per (r_1, r_2)
        keys, local_key, counts = np.unique(r1 * num_r + r2, return_inverse=True, return_counts=True)
        support_keys.append(keys)
        support_counts.append(counts)
        # hits: number of distinct (x, z) per (r_1, r_2) with r(x, z)
        pair_ids, mask = index.lookup_pairs(heads=x, tails=z)
        lhs = scipy.sparse.csr_array(
            (np.ones_like(pair_ids), (local_key[mask], pair_ids)), shape=(len(keys), len(index.pair_keys))
        )
        hits = (lhs @ index.incidence).tocoo()
        hit_keys.append(keys[hits.row])
        hit_relations.append(hits.col)
        hit_counts.append(hits.data)

    # aggregate over chunks; candidates (r_1, r_2) stem from the join, hence always have a non-zero support
    candidates, inverse = np.unique(np.concatenate(support_keys), return_inverse=True)
    support = np.bincount(inverse, weights=np.concatenate(support_counts), minlength=len(candidates)).astype(np.int64)
    hits = scipy.sparse.csr_array(
        (
            np.concatenate(hit_counts),
            (np.searchsorted(candidates, np.concatenate(hit_keys)), np.concatenate(hit_relations)),
        ),
        shape=(len(candidates), num_r),
    )
    first, second = np.divmod(candidates, num_r)
    return _CompositionStatistics(first=first, second=second, support=support, hits=hits)


def _iter_ternary_patterns(
    stats: _CompositionStatistics, relations: np.ndarray, skip_zero: bool = False, batch_size: int = 2**20
) -> Iterable[PatternMatch]:
    """Yield ternary patterns from composition statistics, cf. :func:`iter_ternary_patterns`."""
    logger.debug("Evaluating ternary patterns: {composition}")
    if skip_zero:
        # only the (sparse) non-zero hits; these only occur for occurring relations
        hits = stats.hits.tocoo()
        support = stats.support[hits.row]
        yield from _iter_pattern_matches(PATTERN_TYPE_COMPOSITION, hits.col, support, hits.data / support, skip_zero)
        return
    # each candidate (r_1, r_2) is combined with every relation r
    num_candidates, num_relations = len(stats.support), len(relations)
    step = max(1, batch_size // max(1, num_relations))
    for start in range(0, num_candidates, step):
        stop = min(start + step, num_candidates)
        support = stats.support[start:stop, None]
        confidence = stats.hits[start:stop].toarray()[:, relations] / support
        yield from _iter_pattern_matches(
            PATTERN_TYPE_COMPOSITION,
            np.broadcast_to(relations[None, :], confidence.shape).ravel(),
            np.broadcast_to(support, confidence.shape).ravel(),
            confidence.ravel(),
        )


def iter_ternary_patterns(
    mapped_triples: Collection[tuple[int, int, int]],
) -> Iterable[PatternMatch]:
    r"""
    Yield ternary patterns from ID-based triples.

    ===========  ===========================================
    Pattern      Equation
    ===========  ===========================================
    Composition  $r'(x, y) \land r''(y, z) \implies r(x, z)$
    ===========  ===========================================

    :param mapped_triples:
        A collection of ID-based triples.

    :yields: A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    index = _build_pair_index(mapped_triples)
    yield from _iter_ternary_patterns(_composition_statistics(index), relations=index.relation_order)


def iter_patterns(
    mapped_triples: Collection[tuple[int, int, int]],
    *,
    skip_zero: bool = False,
) -> Iterable[PatternMatch]:
    """Iterate over unary, binary, and ternary patterns.

    :param mapped_triples:
        A collection of ID-based triples.
    :param skip_zero:
        Whether to skip pattern matches with zero confidence.

    :yields: Patterns from :func:`iter_unary_patterns`, func:`iter_binary_patterns`, and :func:`iter_ternary_patterns`.
    """
    index = _build_pair_index(mapped_triples)
    pair_stats = _pair_statistics(index)
    yield from _iter_unary_patterns(pair_stats, skip_zero=skip_zero)
    yield from _iter_binary_patterns(pair_stats, skip_zero=skip_zero)
    yield from _iter_ternary_patterns(
        _composition_statistics(index), relations=index.relation_order, skip_zero=skip_zero
    )


def triple_set_hash(
    mapped_triples: Collection[tuple[int, int, int]],
) -> str:
    """
    Compute an order-invariant hash value for a set of triples given as list of triples.

    :param mapped_triples:
        The ID-based triples.

    :return:
        The hash digest as hex-value string.
    """
    # sort first, for triple order invariance
    return hashlib.sha512("".join(map(str, sorted(mapped_triples))).encode("utf8")).hexdigest()


def _is_injective_mapping(
    df: pd.DataFrame,
    source: str,
    target: str,
) -> tuple[int, float]:
    """
    (Soft-)Determine whether there is an injective mapping from source to target.

    :param df:
        The dataframe.
    :param source:
        The source column.
    :param target:
        The target column.

    :return:
        The number of unique source values, and the relative frequency of unique target per source.
    """
    grouped = df.groupby(by=source)
    support = len(grouped)
    n_unique = grouped.agg({target: "nunique"})[target]
    conf = (n_unique <= 1).mean()
    return support, conf


def iter_relation_cardinality_types(
    mapped_triples: Collection[tuple[int, int, int]],
) -> Iterable[PatternMatch]:
    """Iterate over relation-cardinality types.

    :param mapped_triples:
        A collection of ID-based triples.

    :yields: A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    it = _help_iter_relation_cardinality_types(mapped_triples)
    for relation, support, head_injective_conf, tail_injective_conf in it:
        yield PatternMatch(
            relation,
            CARDINALITY_TYPE_ONE_TO_ONE,
            support,
            head_injective_conf * tail_injective_conf,
        )
        yield PatternMatch(
            relation,
            CARDINALITY_TYPE_ONE_TO_MANY,
            support,
            (1 - head_injective_conf) * tail_injective_conf,
        )
        yield PatternMatch(
            relation,
            CARDINALITY_TYPE_MANY_TO_ONE,
            support,
            head_injective_conf * (1 - tail_injective_conf),
        )
        yield PatternMatch(
            relation,
            CARDINALITY_TYPE_MANY_TO_MANY,
            support,
            (1 - head_injective_conf) * (1 - tail_injective_conf),
        )


def _help_iter_relation_cardinality_types(
    mapped_triples: Collection[tuple[int, int, int]],
) -> Iterable[tuple[int, int, float, float]]:
    df = pd.DataFrame(data=mapped_triples, columns=COLUMN_LABELS)
    for relation, group in df.groupby(by=LABEL_RELATION):
        n_unique_heads, head_injective_conf = _is_injective_mapping(df=group, source=LABEL_HEAD, target=LABEL_TAIL)
        n_unique_tails, tail_injective_conf = _is_injective_mapping(df=group, source=LABEL_TAIL, target=LABEL_HEAD)
        # TODO: what is the support?
        support = n_unique_heads + n_unique_tails
        yield relation, support, head_injective_conf, tail_injective_conf


def _get_skyline(
    xs: Iterable[tuple[int, float]],
) -> Iterable[tuple[int, float]]:
    """Calculate 2-D skyline."""
    # cf. https://stackoverflow.com/questions/19059878/dominant-set-of-points-in-on
    largest_y = float("-inf")
    # sort decreasingly. i dominates j for all j > i in x-dimension
    for x_i, y_i in sorted(xs, reverse=True):
        # if it is also dominated by any y, it is not part of the skyline
        if y_i > largest_y:
            yield x_i, y_i
            largest_y = y_i


def skyline(data_stream: Iterable[PatternMatch]) -> Iterable[PatternMatch]:
    """
    Keep only those entries which are in the support-confidence skyline.

    A pair $(s, c)$ dominates $(s', c')$ if $s > s'$ and $c > c'$. The skyline contains those entries which are not
    dominated by any other entry.

    :param data_stream:
        The stream of data, comprising tuples (relation_id, pattern-type, support, confidence).

    :yields: An entry from the support-confidence skyline.
    """
    # group by (relation id, pattern type)
    data: defaultdict[tuple[int, str], set[tuple[int, float]]] = defaultdict(set)
    for tup in data_stream:
        data[tup[:2]].add(tup[2:])
    # for each group, yield from skyline
    for (r_id, pat), values in data.items():
        for supp, conf in _get_skyline(values):
            yield PatternMatch(r_id, pat, supp, conf)


def _get_counts(
    mapped_triples: MappedTriples,
    column: int | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    unique, counts = mapped_triples[:, column].view(-1).unique(return_counts=True)
    return unique.numpy(), counts.numpy()


def get_entity_counts(
    mapped_triples: MappedTriples,
) -> pd.DataFrame:
    """
    Create a dataframe of entity frequencies.

    :param mapped_triples: shape: (num_triples, 3)
        The mapped triples.

    :return:
        A dataframe with columns ( entity_id | type | count )
    """
    data = []
    for label in (LABEL_HEAD, LABEL_TAIL):
        col = TARGET_TO_INDEX[label]
        unique, counts = _get_counts(mapped_triples=mapped_triples, column=col)
        df = pd.DataFrame(
            {
                ENTITY_ID_COLUMN_NAME: unique,
                COUNT_COLUMN_NAME: counts,
            }
        )
        df[ENTITY_POSITION_COLUMN_NAME] = label
        data.append(df)
    return pd.concat(data, ignore_index=True)


def get_relation_counts(
    mapped_triples: MappedTriples,
) -> pd.DataFrame:
    """
    Create a dataframe of relation frequencies.

    :param mapped_triples: shape: (num_triples, 3)
        The mapped triples.

    :return:
        A dataframe with columns ( relation_id | count )
    """
    return pd.DataFrame(
        data=dict(
            zip(
                [RELATION_ID_COLUMN_NAME, COUNT_COLUMN_NAME],
                _get_counts(mapped_triples=mapped_triples, column=1),
                strict=False,
            )
        )
    )


def relation_pattern_types(
    mapped_triples: Collection[tuple[int, int, int]],
) -> pd.DataFrame:
    r"""
    Categorize relations based on patterns from RotatE [sun2019]_.

    The relation classifications are based upon checking whether the corresponding rules hold with sufficient support
    and confidence. By default, we do not require a minimum support, however, a relatively high confidence.

    The following four non-exclusive classes for relations are considered:

    - symmetry
    - anti-symmetry
    - inversion
    - composition

    This method generally follows the terminology of association rule mining. The patterns are expressed as

    .. math ::

        X_1 \land \cdot \land X_k \implies Y

    where $X_i$ is of the form $r_i(h_i, t_i)$, and some of the $h_i / t_i$ might re-occur in other atoms.
    The *support* of a pattern is the number of distinct instantiations of the variables shared by the left- and
    right-hand side, i.e., the distinct $(x, y)$ pairs for symmetry, anti-symmetry, and inversion, and the distinct
    $(x, z)$ pairs for composition (where the intermediate entity $y$ is projected out).
    The *confidence* is the proportion of these instantiations where the right-hand side is also true.

    :param mapped_triples:
        A collection of ID-based triples.
    :returns:
        A dataframe of relation categorization
    """
    # determine patterns from triples
    # drop zero-confidence
    base = iter_patterns(mapped_triples=mapped_triples, skip_zero=True)

    # keep only skyline
    base = skyline(base)

    # create data frame
    return pd.DataFrame(
        data=list(base),
        columns=[RELATION_ID_COLUMN_NAME, PATTERN_TYPE_COLUMN_NAME, SUPPORT_COLUMN_NAME, CONFIDENCE_COLUMN_NAME],
    ).sort_values(by=[PATTERN_TYPE_COLUMN_NAME, RELATION_ID_COLUMN_NAME, CONFIDENCE_COLUMN_NAME, SUPPORT_COLUMN_NAME])


def relation_injectivity(
    mapped_triples: Collection[tuple[int, int, int]],
    add_labels: bool = True,
    label_to_id: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """
    Calculate "soft" injectivity scores for each relation.

    :param mapped_triples:
        The ID-based triples.
    :param add_labels:
        Whether to add labels.
    :param label_to_id:
        The label to index mapping.

    :return:
        A dataframe with one row per relation, its number of occurrences and head / tail injectivity scores.
    """
    it = _help_iter_relation_cardinality_types(mapped_triples)
    df = pd.DataFrame(
        data=it,
        columns=[RELATION_ID_COLUMN_NAME, SUPPORT_COLUMN_NAME, LABEL_HEAD, LABEL_TAIL],
    )
    return add_relation_labels(df, add_labels=add_labels, label_to_id=label_to_id)


def relation_cardinality_types(
    mapped_triples: Collection[tuple[int, int, int]],
    add_labels: bool = True,
    label_to_id: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    r"""
    Determine the relation cardinality types.

    The possible types are given in relation_cardinality_types.

    .. note ::
        In the current implementation, we have by definition

        .. math ::
            1 = \sum_{type} conf(relation, type)

    .. note ::
       These relation types are also mentioned in [wang2014]_. However, the paper does not provide any details on
       their definition, nor is any code provided. Thus, their exact procedure is unknown and may not coincide with this
       implementation.

    :param mapped_triples:
        The ID-based triples.
    :param add_labels:
        Whether to add relation labels (if available).
    :param label_to_id:
        The label to index mapping.

    :return:
        A dataframe with columns ( relation_id | relation_type )
    """
    # iterate relation types
    base = iter_relation_cardinality_types(mapped_triples=mapped_triples)

    # drop zero-confidence
    base = (pattern for pattern in base if pattern.confidence > 0)

    # keep only skyline
    # does not make much sense, since there is always exactly one entry per (relation, pattern) pair
    # base = skyline(base)

    # create data frame
    df = pd.DataFrame(
        data=base,
        columns=[
            RELATION_ID_COLUMN_NAME,
            CARDINALITY_TYPE_COLUMN_NAME,
            SUPPORT_COLUMN_NAME,
            CONFIDENCE_COLUMN_NAME,
        ],
    )
    return add_relation_labels(df, add_labels=add_labels, label_to_id=label_to_id)


def entity_relation_co_occurrence(
    mapped_triples: MappedTriples,
) -> pd.DataFrame:
    """
    Calculate entity-relation co-occurrence.

    :param mapped_triples:
        The ID-based triples.

    :return:
        A dataframe with columns ( entity_id | relation_id | type | count )
    """
    data = []

    for name, columns in {
        LABEL_HEAD: (COLUMN_HEAD, COLUMN_RELATION),
        LABEL_TAIL: (COLUMN_TAIL, COLUMN_RELATION),
    }.items():
        unique, counts = mapped_triples[:, columns].unique(dim=0, return_counts=True)
        e, r = unique.t().numpy()
        df = pd.DataFrame(
            data={
                ENTITY_ID_COLUMN_NAME: e,
                RELATION_ID_COLUMN_NAME: r,
                COUNT_COLUMN_NAME: counts.numpy(),
            },
        )
        df[ENTITY_POSITION_COLUMN_NAME] = name
        data.append(df)

    return pd.concat(data, ignore_index=True)


def get_relation_functionality(
    mapped_triples: Collection[tuple[int, int, int]],
    add_labels: bool = True,
    label_to_id: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Calculate relation functionalities.

    :param mapped_triples:
        The ID-based triples.
    :param add_labels:
        Should the labels be added to the dataframe?
    :param label_to_id:
        The label to index mapping.

    :return:
        A dataframe with columns ( functionality | inverse_functionality )
    """
    df = pd.DataFrame(data=mapped_triples, columns=COLUMN_LABELS)
    df = df.groupby(by=LABEL_RELATION).agg({LABEL_HEAD: ["nunique", COUNT_COLUMN_NAME], LABEL_TAIL: "nunique"})
    df[FUNCTIONALITY_COLUMN_NAME] = df[(LABEL_HEAD, "nunique")] / df[(LABEL_HEAD, COUNT_COLUMN_NAME)]
    df[INVERSE_FUNCTIONALITY_COLUMN_NAME] = df[(LABEL_TAIL, "nunique")] / df[(LABEL_HEAD, COUNT_COLUMN_NAME)]
    df = df[[FUNCTIONALITY_COLUMN_NAME, INVERSE_FUNCTIONALITY_COLUMN_NAME]]
    df.columns = df.columns.droplevel(1)
    df.index.name = RELATION_ID_COLUMN_NAME
    df = df.reset_index()
    return add_relation_labels(df, add_labels=add_labels, label_to_id=label_to_id)
