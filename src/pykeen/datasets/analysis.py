# -*- coding: utf-8 -*-

"""Dataset analysis utilities."""

import hashlib
import itertools as itt
import logging
from collections import defaultdict
from typing import Collection, DefaultDict, Iterable, Mapping, NamedTuple, Optional, Set, Tuple, Union

import numpy
import pandas as pd
import torch
from tqdm import tqdm

from .base import Dataset
from ..constants import PYKEEN_DATASETS
from ..utils import invert_mapping

logger = logging.getLogger(__name__)

__all__ = [
    'get_id_counts',
    'relation_classification',
    'relation_count_dataframe',
    'entity_count_dataframe',
    'entity_relation_co_occurrence_dataframe',
    'skyline',
    'composition_candidates',
    'iter_patterns',
    'iter_unary_patterns',
    'iter_binary_patterns',
    'iter_ternary_patterns',
    'triple_set_hash',
]

SUBSET_LABELS = ('testing', 'training', 'validation', 'total')


# PatternMatch = namedtuple('PatternMatch', ['relation_id', 'pattern_type', 'support', 'confidence'])

class PatternMatch(NamedTuple):
    """A pattern match tuple of relation_id, pattern_type, support, and confidence."""

    relation_id: int
    pattern_type: str
    support: int
    confidence: float


def get_id_counts(id_tensor: torch.LongTensor, num_ids: int) -> numpy.ndarray:
    """Create a dense tensor of ID counts.

    :param id_tensor:
        The tensor of IDs.
    :param num_ids:
        The number of IDs.

    :return: shape: (num_ids,)
         The counts for each individual ID from {0, 1, ..., num_ids-1}.
    """
    unique, counts = id_tensor.unique(return_counts=True)
    total_counts = numpy.zeros(shape=(num_ids,), dtype=numpy.int64)
    total_counts[unique.numpy()] = counts.numpy()
    return total_counts


def relation_count_dataframe(dataset: Dataset) -> pd.DataFrame:
    """Create a dataframe with relation counts for all subsets, and the full dataset.

    Example usage:

    >>> from pykeen.datasets import Nations
    >>> dataset = Nations()
    >>> from pykeen.datasets.analysis import relation_count_dataframe
    >>> df = relation_count_dataframe(dataset=dataset)

    # Get the most frequent relations in training
    >>> df.sort_values(by="training").head()

    # Get all relations which do not occur in the test part
    >>> df[df["testing"] == 0]

    :param dataset:
        The dataset.

    :return:
        A dataframe with one row per relation.
    """
    data = {
        subset_name: get_id_counts(
            id_tensor=triples_factory.mapped_triples[:, 1],
            num_ids=dataset.num_relations,
        )
        for subset_name, triples_factory in dataset.factory_dict.items()
    }
    data['total'] = sum(data.values())
    index = sorted(dataset.relation_to_id, key=dataset.relation_to_id.get)
    df = pd.DataFrame(data=data, index=index, columns=SUBSET_LABELS)
    df.index.name = 'relation_label'
    return df


def entity_count_dataframe(dataset: Dataset) -> pd.DataFrame:
    """Create a dataframe with head/tail/both counts for all subsets, and the full dataset.

    Example usage:

    >>> from pykeen.datasets import FB15k237
    >>> dataset = FB15k237()
    >>> from pykeen.datasets.analysis import relation_count_dataframe
    >>> df = entity_count_dataframe(dataset=dataset)

    # Get the most frequent entities in training (counting both, occurrences as heads as well as occurences as tails)
    >>> df.sort_values(by=[("training", "total")]).tail()

    # Get entities which do not occur in testing
    >>> df[df[("testing", "total")] == 0]

    # Get entities which never occur as head entity (in any subset)
    >>> df[df[("total", "head")] == 0]

    :param dataset:
        The dataset.

    :return:
        A dataframe with one row per entity.
    """
    data = {}
    num_entities = dataset.num_entities
    second_level_order = ('head', 'tail', 'total')
    for subset_name, triples_factory in dataset.factory_dict.items():
        for col, col_name in zip((0, 2), ('head', 'tail')):
            data[subset_name, col_name] = get_id_counts(
                id_tensor=triples_factory.mapped_triples[:, col],
                num_ids=num_entities,
            )
        data[subset_name, 'total'] = data[subset_name, 'head'] + data[subset_name, 'tail']
    for kind in ('head', 'tail', 'total'):
        data['total', kind] = sum(data[subset_name, kind] for subset_name in dataset.factory_dict.keys())
    index = sorted(dataset.entity_to_id, key=dataset.entity_to_id.get)
    df = pd.DataFrame(
        data=data,
        index=index,
        columns=pd.MultiIndex.from_product(iterables=[SUBSET_LABELS, second_level_order]),
    )
    df.index.name = 'entity_label'
    return df


def entity_relation_co_occurrence_dataframe(dataset: Dataset) -> pd.DataFrame:
    """Create a dataframe of entity/relation co-occurrence.

    This information can be seen as a form of pseudo-typing, e.g. entity A is something which can be a head of
    `born_in`.

    Example usages:
    >>> from pykeen.datasets import Nations
    >>> dataset = Nations()
    >>> from pykeen.datasets.analysis import relation_count_dataframe
    >>> df = entity_count_dataframe(dataset=dataset)

    # Which countries have to most embassies (considering only training triples)?
    >>> df.loc['training', ('head', 'embassy')].sort_values().tail()

    # In which countries are to most embassies (considering only training triples)?
    >>> df.loc['training', ('tail', 'embassy')].sort_values().tail()

    :param dataset:
        The dataset.

    :return:
        A dataframe with a multi-index (subset, entity_id) as index, and a multi-index (kind, relation) as columns,
        where subset in {'training', 'validation', 'testing', 'total'}, and kind in {'head', 'tail'}. For each entity,
        the corresponding row can be seen a pseudo-type, i.e. for which relations it may occur as head/tail.
    """
    num_relations = dataset.num_relations
    num_entities = dataset.num_entities
    data = numpy.zeros(shape=(4 * num_entities, 2 * num_relations), dtype=numpy.int64)
    for i, (_, triples_factory) in enumerate(sorted(dataset.factory_dict.items())):
        # head-relation co-occurrence
        unique_hr, counts_hr = triples_factory.mapped_triples[:, :2].unique(dim=0, return_counts=True)
        h, r = unique_hr.t().numpy()
        data[i * num_entities:(i + 1) * num_entities, :num_relations][h, r] = counts_hr.numpy()

        # tail-relation co-occurrence
        unique_rt, counts_rt = triples_factory.mapped_triples[:, 1:].unique(dim=0, return_counts=True)
        r, t = unique_rt.t().numpy()
        data[i * num_entities:(i + 1) * num_entities, num_relations:][t, r] = counts_rt.numpy()

    # full dataset
    data[3 * num_entities:] = sum(data[i * num_entities:(i + 1) * num_entities] for i in range(3))
    entity_id_to_label, relation_id_to_label = [
        invert_mapping(mapping=mapping)
        for mapping in (dataset.entity_to_id, dataset.relation_to_id)
    ]
    return pd.DataFrame(
        data=data,
        index=pd.MultiIndex.from_product([
            sorted(dataset.factory_dict.keys()) + ['total'],
            [entity_id_to_label[entity_id] for entity_id in range(num_entities)],
        ]),
        columns=pd.MultiIndex.from_product([
            ('head', 'tail'),
            [relation_id_to_label[relation_id] for relation_id in range(num_relations)],
        ]),
    )


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
    # index triples
    # incoming relations per entity
    ins: DefaultDict[int, Set[int]] = defaultdict(set)
    # outgoing relations per entity
    outs: DefaultDict[int, Set[int]] = defaultdict(set)
    for h, r, t in mapped_triples:
        outs[h].add(r)
        ins[t].add(r)

    # return candidates
    return {
        (r1, r2)
        for e, r1s in ins.items()
        for r1, r2 in itt.product(r1s, outs[e])
    }


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

    :param pairs:
        A mapping from relations to the set of entity pairs.

    :yields: A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    logger.debug("Evaluating unary patterns: {symmetry, anti-symmetry}")
    for r, ht in pairs.items():
        support = len(ht)
        rev_ht = {(t, h) for h, t in ht}
        confidence = len(ht.intersection(rev_ht)) / support
        yield PatternMatch(r, "symmetry", support, confidence)
        confidence = len(ht.difference(rev_ht)) / support
        yield PatternMatch(r, "anti-symmetry", support, 1 - confidence)


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
        yield PatternMatch(r, "inversion", support, confidence)


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
    # indexing triples for fast join r1 & r2
    adj: DefaultDict[int, DefaultDict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    for h, r, t in mapped_triples:
        adj[r][h].add(t)
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
            for z in adj[r2]
        }
        support = len(lhs)
        # skip empty support
        # TODO: Can this happen after pre-filtering?
        if not support:
            continue
        for r, ht in pairs.items():
            confidence = len(lhs.intersection(ht)) / support
            yield PatternMatch(r, "composition", support, confidence)


def iter_patterns(
    mapped_triples: Collection[Tuple[int, int, int]],
) -> Iterable[PatternMatch]:
    """Iterate over unary, binary, and ternary patterns.

    :param mapped_triples:
        A collection of ID-based triples.

    :yields: Patterns from :func:`iter_unary_patterns`, func:`iter_binary_patterns`, and :func:`iter_ternary_patterns`.
    """
    # indexing triples for fast lookup of entity pair sets
    pairs: DefaultDict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for h, r, t in mapped_triples:
        pairs[r].add((h, t))

    yield from iter_unary_patterns(pairs=pairs)
    yield from iter_binary_patterns(pairs=pairs)
    yield from iter_ternary_patterns(mapped_triples, pairs=pairs)


def triple_set_hash(mapped_triples: Collection[Tuple[int, int, int]]):
    """
    Compute an order-invariant hash value for a set of triples given as tensor.

    :param mapped_triples:
        The ID-based triples.

    :return:
        The hash object.
    """
    return hashlib.sha512("".join(map(str, sorted(mapped_triples))).encode("utf8"))


def relation_classification(
    dataset: Dataset,
    min_support: int = 0,
    min_confidence: float = 0.95,
    drop_confidence: bool = True,
    parts: Optional[Collection[str]] = None,
    force: bool = False,
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

    :param dataset:
        The dataset to investigate.
    :param min_support:
        A minimum support for patterns.
    :param min_confidence:
        A minimum confidence for the tested patterns.
    :param drop_confidence:
        Whether to drop the support/confidence information from the result frame, and also drop duplicates.
    :param parts:
        Only use certain parts of the dataset, e.g., train triples. Defaults to using all triples, i.e.
        {"training", "validation", "testing}.
    :param force:
        Whether to enforce re-calculation even if a cached version is available.

    .. warning ::

        If you intend to use the relation categorization as input to your model, or hyper-parameter selection, do *not*
        include testing triples to avoid leakage!

    :return:
        A dataframe with columns {"relation_id", "pattern", "support"?, "confidence"?}.
    """
    parts = _normalize_parts(dataset, parts)
    mapped_triples = _get_mapped_triples(dataset, parts)

    # include hash over triples into cache-file name
    # sort first, for triple order invariance
    ph = triple_set_hash(mapped_triples).hexdigest()[:16]

    # include part hash into cache-file name
    cache_path = PYKEEN_DATASETS.joinpath(dataset.__class__.__name__.lower(), f"relation_patterns_{ph}.tsv.xz")

    # re-use cached file if possible
    if not cache_path.is_file() or force:
        # select triples
        mapped_triples = torch.cat([
            dataset.factory_dict[part].mapped_triples
            for part in parts
        ], dim=0)

        # determine patterns from triples
        base = iter_patterns(mapped_triples=mapped_triples.tolist())

        # drop zero-confidence
        base = (
            pattern
            for pattern in base
            if pattern.confidence > 0
        )

        # keep only skyline
        base = skyline(base)

        # create data frame
        df = pd.DataFrame(
            data=list(base),
            columns=["relation_id", "pattern", "support", "confidence"],
        ).sort_values(by=["pattern", "relation_id", "confidence", "support"])

        # save to file
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(cache_path, sep="\t", index=False)
        logger.info(f"Cached {len(df)} relational pattern entries to {cache_path.as_uri()}")
    else:
        df = pd.read_csv(cache_path, sep="\t")
        logger.info(f"Loaded {len(df)} precomputed relational patterns from {cache_path.as_uri()}")

    # Prune by support and confidence
    df = df[(df["support"] >= min_support) & (df["confidence"] >= min_confidence)]

    if drop_confidence:
        df = df[["relation_id", "pattern"]].drop_duplicates()

    return df


def _get_mapped_triples(dataset: Dataset, parts: Collection[str]) -> Collection[Tuple[int, int, int]]:
    return torch.cat([
        dataset.factory_dict[part].mapped_triples
        for part in parts
    ], dim=0).tolist()


def _normalize_parts(dataset: Dataset, parts: Union[None, str, Collection[str]]) -> Collection[str]:
    if parts is None:
        parts = dataset.factory_dict.keys()
    elif isinstance(parts, str):
        parts = [parts]
    return parts


def relation_classification2(
    *,
    dataset: Dataset,
    parts: Optional[Collection[str]] = None,
):
    """
    Determine whether relations are of type 1:1, 1:n, m:1, or m:n.

    :param dataset:
        The dataset to investigate.
    :param parts:
        Only use certain parts of the dataset, e.g., train triples. Defaults to using all triples, i.e.
        {"training", "validation", "testing}.

    :return:
        A dataframe with columns ( relation_id | relation_type )
    """
    # TODO: This function needs a better name
    # TODO: Consider merging with other analysis methods
    parts = _normalize_parts(dataset=dataset, parts=parts)
    mapped_triples = _get_mapped_triples(dataset=dataset, parts=parts)
    df = pd.DataFrame(data=mapped_triples, columns=["h", "r", "t"])
    data = []
    for relation, group in df.groupby(by="r"):
        maximum_t_per_h = group.groupby(by="h").agg({"t": "nunique"})["t"].max()
        maximum_h_per_t = group.groupby(by="t").agg({"h": "nunique"})["h"].max()
        if maximum_h_per_t > 1 and maximum_t_per_h > 1:
            relation_type = "m:n"
        elif maximum_h_per_t > 1:  # and maximum_t_per_h == 1
            relation_type = "m:1"
        elif maximum_t_per_h > 1:  # and maximum_h_per_t == 1
            relation_type = "1:n"
        else:
            relation_type = "1:1"
        data.append((relation, relation_type))
    return pd.DataFrame(data=data, columns=["relation_id", "relation_type"])
