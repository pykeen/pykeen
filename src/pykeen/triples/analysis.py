# -*- coding: utf-8 -*-

"""Analysis utilities for (mapped) triples."""

import hashlib
import itertools as itt
import logging
from collections import defaultdict
from typing import Collection, DefaultDict, Iterable, Mapping, NamedTuple, Optional, Sequence, Set, Tuple, Union

import numpy
import pandas as pd
from tqdm.auto import tqdm

from . import TriplesFactory
from ..typing import MappedTriples

logger = logging.getLogger(__name__)

__all__ = [
    "add_entity_labels",
    "add_relation_labels",
    "create_relation_to_entity_set_mapping",
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
POSITION_TAIL = "tail"
POSITION_HEAD = "head"


def _add_labels(
    df: pd.DataFrame,
    add_labels: bool,
    label_to_id: Optional[Mapping[str, int]],
    id_column: str,
    label_column: str,
    label_to_id_mapping_name: str,
    triples_factory: Optional[TriplesFactory] = None,
) -> pd.DataFrame:
    """Add labels to a dataframe."""
    if not add_labels:
        return df
    if not label_to_id:
        if not triples_factory:
            raise ValueError
        label_to_id = getattr(triples_factory, label_to_id_mapping_name)
    assert label_to_id is not None
    return pd.merge(
        left=df,
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
    label_to_id: Optional[Mapping[str, int]] = None,
    triples_factory: Optional[TriplesFactory] = None,
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
    label_to_id: Optional[Mapping[str, int]] = None,
    triples_factory: Optional[TriplesFactory] = None,
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


def composition_candidates(
    mapped_triples: Iterable[Tuple[int, int, int]],
) -> Collection[Tuple[int, int]]:
    r"""Pre-filtering relation pair candidates for composition pattern.

    Determines all relation pairs $(r, r')$ with at least one entity $e$ such that

    .. math ::

        \{(h, r, e), (e, r', t)\} \subseteq \mathcal{T}

    :param mapped_triples:
        An iterable over ID-based triples. Only consumed once.

    :return:
        A set of relation pairs.
    """
    ins, outs = index_relations(mapped_triples)

    # return candidates
    return {
        (r1, r2)
        for tail, tail_in_relations in ins.items()
        if tail in outs
        for r1, r2 in itt.product(tail_in_relations, outs[tail])
    }


def index_relations(
    triples: Iterable[Tuple[int, int, int]],
) -> Tuple[Mapping[int, Set[int]], Mapping[int, Set[int]]]:
    """Create an index for in-relationships and out-relationships."""
    # index triples
    # incoming relations per entity
    ins: DefaultDict[int, Set[int]] = defaultdict(set)
    # outgoing relations per entity
    outs: DefaultDict[int, Set[int]] = defaultdict(set)
    for h, r, t in triples:
        outs[h].add(r)
        ins[t].add(r)
    return dict(ins), dict(outs)


def create_relation_to_entity_set_mapping(
    triples: Iterable[Tuple[int, int, int]],
) -> Tuple[Mapping[int, Set[int]], Mapping[int, Set[int]]]:
    """
    Create mappings from relation IDs to the set of their head / tail entities.

    :param triples:
        The triples.

    :return:
        A pair of dictionaries, each mapping relation IDs to entity ID sets.
    """
    tails = defaultdict(set)
    heads = defaultdict(set)
    for h, r, t in triples:
        heads[r].add(h)
        tails[r].add(t)
    return heads, tails


def index_pairs(triples: Iterable[Tuple[int, int, int]]) -> Mapping[int, Set[Tuple[int, int]]]:
    """Create a mapping from relation to head/tail pairs for fast lookup."""
    rv: DefaultDict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for h, r, t in triples:
        rv[r].add((h, t))
    return dict(rv)


def get_adjacency_dict(triples: Iterable[Tuple[int, int, int]]) -> Mapping[int, Mapping[int, Set[int]]]:
    """Create a two-level mapping from relation to head to tails."""
    # indexing triples for fast join r1 & r2
    rv: DefaultDict[int, DefaultDict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    for h, r, t in triples:
        rv[r][h].add(t)
    return rv


def iter_unary_patterns(
    pairs: Mapping[int, Set[Tuple[int, int]]],
) -> Iterable[PatternMatch]:
    r"""
    Yield unary patterns from pre-indexed triples.

    =============  ===============================
    Pattern        Equation
    =============  ===============================
    Symmetry       $r(x, y) \implies r(y, x)$
    Anti-Symmetry  $r(x, y) \implies \neg r(y, x)$
    =============  ===============================

    .. note ::
        By definition, we have confidence(anti-symmetry) = 1 - confidence(symmetry).

    :param pairs:
        A mapping from relations to the set of entity pairs.

    :yields: A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    logger.debug("Evaluating unary patterns: {symmetry, anti-symmetry}")
    for r, ht in pairs.items():
        support = len(ht)
        rev_ht = {(t, h) for h, t in ht}
        confidence = len(ht.intersection(rev_ht)) / support
        yield PatternMatch(r, PATTERN_TYPE_SYMMETRY, support, confidence)
        # confidence = len(ht.difference(rev_ht)) / support = 1 - len(ht.intersection(rev_ht)) / support
        yield PatternMatch(r, PATTERN_TYPE_ANTI_SYMMETRY, support, 1 - confidence)


def iter_binary_patterns(
    pairs: Mapping[int, Set[Tuple[int, int]]],
) -> Iterable[PatternMatch]:
    r"""
    Yield binary patterns from pre-indexed triples.

    =========  ===========================
    Pattern    Equation
    =========  ===========================
    Inversion  $r'(x, y) \implies r(y, x)$
    =========  ===========================

    :param pairs:
        A mapping from relations to the set of entity pairs.

    :yields: A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    logger.debug("Evaluating binary patterns: {inversion}")
    for (_r1, ht1), (r, ht2) in itt.combinations(pairs.items(), r=2):
        support = len(ht1)
        confidence = len(ht1.intersection(ht2)) / support
        yield PatternMatch(r, PATTERN_TYPE_INVERSION, support, confidence)


def iter_ternary_patterns(
    mapped_triples: Collection[Tuple[int, int, int]],
    pairs: Mapping[int, Set[Tuple[int, int]]],
) -> Iterable[PatternMatch]:
    r"""
    Yield ternary patterns from pre-indexed triples.

    ===========  ===========================================
    Pattern      Equation
    ===========  ===========================================
    Composition  $r'(x, y) \land r''(y, z) \implies r(x, z)$
    ===========  ===========================================

    :param mapped_triples:
        A collection of ID-based triples.
    :param pairs:
        A mapping from relations to the set of entity pairs.

    :yields: A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    logger.debug("Evaluating ternary patterns: {composition}")
    # composition r1(x, y) & r2(y, z) => r(x, z)
    adj = get_adjacency_dict(mapped_triples)

    # actual evaluation of the pattern
    for r1, r2 in tqdm(
        composition_candidates(mapped_triples),
        desc="Checking ternary patterns",
        unit="pattern",
        unit_scale=True,
    ):
        lhs = {
            (x, z)
            for x, y in pairs[r1]
            for z in adj[r2][y]
        }
        support = len(lhs)
        # skip empty support
        # TODO: Can this happen after pre-filtering?
        if not support:
            continue
        for r, ht in pairs.items():
            confidence = len(lhs.intersection(ht)) / support
            yield PatternMatch(r, PATTERN_TYPE_COMPOSITION, support, confidence)


def iter_patterns(
    mapped_triples: Collection[Tuple[int, int, int]],
) -> Iterable[PatternMatch]:
    """Iterate over unary, binary, and ternary patterns.

    :param mapped_triples:
        A collection of ID-based triples.

    :yields: Patterns from :func:`iter_unary_patterns`, func:`iter_binary_patterns`, and :func:`iter_ternary_patterns`.
    """
    pairs = index_pairs(mapped_triples)

    yield from iter_unary_patterns(pairs=pairs)
    yield from iter_binary_patterns(pairs=pairs)
    yield from iter_ternary_patterns(mapped_triples, pairs=pairs)


def triple_set_hash(
    mapped_triples: Collection[Tuple[int, int, int]],
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
) -> Tuple[int, float]:
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
    mapped_triples: Collection[Tuple[int, int, int]],
) -> Iterable[PatternMatch]:
    """Iterate over relation-cardinality types.

    :param mapped_triples:
        A collection of ID-based triples.

    :yields: A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    it = _help_iter_relation_cardinality_types(mapped_triples)
    for relation, support, head_injective_conf, tail_injective_conf in it:
        yield PatternMatch(
            relation, CARDINALITY_TYPE_ONE_TO_ONE, support, head_injective_conf * tail_injective_conf,
        )
        yield PatternMatch(
            relation, CARDINALITY_TYPE_ONE_TO_MANY, support, (1 - head_injective_conf) * tail_injective_conf,
        )
        yield PatternMatch(
            relation, CARDINALITY_TYPE_MANY_TO_ONE, support, head_injective_conf * (1 - tail_injective_conf),
        )
        yield PatternMatch(
            relation, CARDINALITY_TYPE_MANY_TO_MANY, support, (1 - head_injective_conf) * (1 - tail_injective_conf),
        )


def _help_iter_relation_cardinality_types(
    mapped_triples: Collection[Tuple[int, int, int]],
) -> Iterable[Tuple[int, int, float, float]]:
    df = pd.DataFrame(data=mapped_triples, columns=["h", "r", "t"])
    for relation, group in df.groupby(by="r"):
        n_unique_heads, head_injective_conf = _is_injective_mapping(df=group, source="h", target="t")
        n_unique_tails, tail_injective_conf = _is_injective_mapping(df=group, source="t", target="h")
        # TODO: what is the support?
        support = n_unique_heads + n_unique_tails
        yield relation, support, head_injective_conf, tail_injective_conf


def _get_skyline(
    xs: Iterable[Tuple[int, float]],
) -> Iterable[Tuple[int, float]]:
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
    data: DefaultDict[Tuple[int, str], Set[Tuple[int, float]]] = defaultdict(set)
    for tup in data_stream:
        data[tup[:2]].add(tup[2:])
    # for each group, yield from skyline
    for (r_id, pat), values in data.items():
        for supp, conf in _get_skyline(values):
            yield PatternMatch(r_id, pat, supp, conf)


def _get_counts(
    mapped_triples: MappedTriples,
    column: Union[int, Sequence[int]],
) -> Tuple[numpy.ndarray, numpy.ndarray]:
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
    for label, col in (
        (POSITION_HEAD, 0),
        (POSITION_TAIL, 2),
    ):
        unique, counts = _get_counts(mapped_triples=mapped_triples, column=col)
        df = pd.DataFrame({
            ENTITY_ID_COLUMN_NAME: unique,
            COUNT_COLUMN_NAME: counts,
        })
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
    return pd.DataFrame(data=dict(zip(
        [RELATION_ID_COLUMN_NAME, COUNT_COLUMN_NAME],
        _get_counts(mapped_triples=mapped_triples, column=1),
    )))


def relation_pattern_types(
    mapped_triples: Collection[Tuple[int, int, int]],
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
    The *support* of a pattern is the number of distinct instantiations of all variables for the left hand side.
    The *confidence* is the proportion of these instantiations where the right-hand side is also true.
    """
    # determine patterns from triples
    base = iter_patterns(mapped_triples=mapped_triples)

    # drop zero-confidence
    base = (
        pattern
        for pattern in base
        if pattern.confidence > 0
    )

    # keep only skyline
    base = skyline(base)

    # create data frame
    return pd.DataFrame(
        data=list(base),
        columns=[RELATION_ID_COLUMN_NAME, PATTERN_TYPE_COLUMN_NAME, SUPPORT_COLUMN_NAME, CONFIDENCE_COLUMN_NAME],
    ).sort_values(by=[PATTERN_TYPE_COLUMN_NAME, RELATION_ID_COLUMN_NAME, CONFIDENCE_COLUMN_NAME, SUPPORT_COLUMN_NAME])


def relation_injectivity(
    mapped_triples: Collection[Tuple[int, int, int]],
    add_labels: bool = True,
    label_to_id: Optional[Mapping[str, int]] = None,
) -> pd.DataFrame:
    """
    Calculate "soft" injectivity scores for each relation.

    :param mapped_triples:
        The ID-based triples.
    :param add_labels:
        Whether to add labels.
    :param label_to_id:
        The label to Id mapping.

    :return:
        A dataframe with one row per relation, its number of occurrences and head / tail injectivity scores.
    """
    it = _help_iter_relation_cardinality_types(mapped_triples)
    df = pd.DataFrame(
        data=it,
        columns=[RELATION_ID_COLUMN_NAME, SUPPORT_COLUMN_NAME, POSITION_HEAD, POSITION_TAIL],
    )
    return add_relation_labels(df, add_labels=add_labels, label_to_id=label_to_id)


def relation_cardinality_types(
    mapped_triples: Collection[Tuple[int, int, int]],
    add_labels: bool = True,
    label_to_id: Optional[Mapping[str, int]] = None,
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

    :return:
        A dataframe with columns ( relation_id | relation_type )
    """
    # iterate relation types
    base = iter_relation_cardinality_types(mapped_triples=mapped_triples)

    # drop zero-confidence
    base = (
        pattern
        for pattern in base
        if pattern.confidence > 0
    )

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

    for name, columns in dict(
        head=[0, 1],
        tail=[2, 1],
    ).items():
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
    mapped_triples: Collection[Tuple[int, int, int]],
    add_labels: bool = True,
    label_to_id: Optional[Mapping[str, int]] = None,
) -> pd.DataFrame:
    """Calculate relation functionalities.

    :param mapped_triples:
        The ID-based triples.

    :return:
        A dataframe with columns ( functionality | inverse_functionality )
    """
    df = pd.DataFrame(data=mapped_triples, columns=["h", "r", "t"])
    df = df.groupby(by="r").agg(dict(
        h=["nunique", COUNT_COLUMN_NAME],
        t="nunique",
    ))
    df[FUNCTIONALITY_COLUMN_NAME] = df[("h", "nunique")] / df[("h", COUNT_COLUMN_NAME)]
    df[INVERSE_FUNCTIONALITY_COLUMN_NAME] = df[("t", "nunique")] / df[("h", COUNT_COLUMN_NAME)]
    df = df[[FUNCTIONALITY_COLUMN_NAME, INVERSE_FUNCTIONALITY_COLUMN_NAME]]
    df.columns = df.columns.droplevel(1)
    df.index.name = RELATION_ID_COLUMN_NAME
    df = df.reset_index()
    return add_relation_labels(df, add_labels=add_labels, label_to_id=label_to_id)
