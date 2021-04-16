# -*- coding: utf-8 -*-

"""Dataset analysis utilities."""
import hashlib
import itertools as itt
import logging
from collections import defaultdict
from operator import itemgetter
from typing import Collection, DefaultDict, Iterable, Mapping, NamedTuple, Optional, Set, Tuple

import numpy
import pandas
import torch
from tqdm import tqdm

from .base import Dataset
from ..constants import PYKEEN_DATASETS
from ..utils import invert_mapping

logger = logging.getLogger(__name__)

SUBSET_LABELS = ('testing', 'training', 'validation', 'total')


class PatternMatch(NamedTuple):
    """A pattern match tuple of relation_id, pattern_type, support, and confidence."""

    relation_id: int
    pattern_type: str
    support: int
    confidence: float


def get_id_counts(
    id_tensor: torch.LongTensor,
    num_ids: int,
) -> numpy.ndarray:
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


def relation_count_dataframe(dataset: Dataset) -> pandas.DataFrame:
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
    data = dict()
    for subset_name, triples_factory in dataset.factory_dict.items():
        data[subset_name] = get_id_counts(
            id_tensor=triples_factory.mapped_triples[:, 1],
            num_ids=dataset.num_relations,
        )
    data['total'] = sum(data[subset_name] for subset_name in dataset.factory_dict.keys())
    index = [relation_label for (relation_label, _) in sorted(dataset.relation_to_id.items(), key=itemgetter(1))]
    df = pandas.DataFrame(data=data, index=index, columns=SUBSET_LABELS)
    df.index.name = 'relation_label'
    return df


def entity_count_dataframe(dataset: Dataset) -> pandas.DataFrame:
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
    index = [entity_label for (entity_label, _) in sorted(dataset.entity_to_id.items(), key=itemgetter(1))]
    df = pandas.DataFrame(
        data=data,
        index=index,
        columns=pandas.MultiIndex.from_product(iterables=[SUBSET_LABELS, second_level_order]),
    )
    df.index.name = 'entity_label'
    return df


def entity_relation_co_occurrence_dataframe(dataset: Dataset) -> pandas.DataFrame:
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
    for i, (_subset_name, triples_factory) in enumerate(sorted(dataset.factory_dict.items())):
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
    return pandas.DataFrame(
        data=data,
        index=pandas.MultiIndex.from_product([
            sorted(dataset.factory_dict.keys()) + ['total'],
            [entity_id_to_label[entity_id] for entity_id in range(num_entities)],
        ]),
        columns=pandas.MultiIndex.from_product([
            ('head', 'tail'),
            [relation_id_to_label[relation_id] for relation_id in range(num_relations)],
        ]),
    )


def _get_skyline(xs: Collection[Tuple[int, float]]) -> Collection[Tuple[int, float]]:
    # TODO: naive implementation, O(n2)
    return {
        (s, c)
        for s, c in xs
        if not any(
            s2 > s and c2 > c
            for s2, c2 in xs
        )
    }


def skyline(data_stream: Iterable[PatternMatch]) -> Iterable[PatternMatch]:
    """
    Keep only those entries which are in the support-confidence skyline.

    A pair $(s, c)$ dominates $(s', c')$ if $s > s'$ and $c > c'$. The skyline contains those entries which are not
    dominated by any other entry.

    :param data_stream:
        The stream of data, comprising tuples (relation_id, pattern-type, support, confidence).

    :yields:
        An entry from the support-confidence skyline.
    """
    # group by (relation id, pattern type)
    data: DefaultDict[Tuple[int, str], Set[Tuple[int, float]]] = defaultdict(set)
    for tup in data_stream:
        data[tup[:2]].add(tup[2:])
    # for each group, yield from skyline
    for (r_id, pat), values in data.items():
        yield from (
            PatternMatch(r_id, pat, supp, conf)
            for supp, conf in _get_skyline(values)
        )


def _composition_candidates(
    mapped_triples_list: Collection[Tuple[int, int, int]],
) -> Collection[Tuple[int, int]]:
    r"""
    Pre-filtering relation pair candidates for composition pattern.

    Determines all relation pairs $(r, r')$ with at least one entity e such that

    .. math ::

        \{(h, r, e), (e, r', t)\} \subseteq \mathcal{T}

    :param mapped_triples_list:
        The collection of ID-based triples.

    :return:
        A set of relation pairs.
    """
    # index triples
    # incoming relations per entity
    ins: DefaultDict[int, Set[int]] = defaultdict(set)
    # outgoing relations per entity
    outs: DefaultDict[int, Set[int]] = defaultdict(set)
    for h, r, t in mapped_triples_list:
        outs[h].add(r)
        ins[t].add(r)

    # return candidates
    return {
        (r1, r2)
        for e, r1s in ins.items()
        for r1 in r1s
        for r2 in outs[e]
    }


def yield_unary_patterns(
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

    :yields:
        A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    logger.debug("Evaluating unary patterns: {symmetry, anti-symmetry}")
    for r, ht in pairs.items():
        support = len(ht)
        rev_ht = {(t, h) for h, t in ht}
        confidence = len(ht.intersection(rev_ht)) / support
        yield PatternMatch(r, "symmetry", support, confidence)
        confidence = len(ht.difference(rev_ht)) / support
        yield PatternMatch(r, "anti-symmetry", support, 1 - confidence)


def yield_binary_patterns(
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

    :yields:
        A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    logger.debug("Evaluating binary patterns: {inversion}")
    for (r1, ht1), (r, ht2) in itt.combinations(pairs.items(), r=2):
        support = len(ht1)
        confidence = len(ht1.intersection(ht2)) / support
        yield PatternMatch(r, "inversion", support, confidence)


def yield_ternary_patterns(
    mapped_triples_list: Collection[Tuple[int, int, int]],
    pairs: Mapping[int, Set[Tuple[int, int]]],
) -> Iterable[PatternMatch]:
    r"""
    Yield ternary patterns from pre-indexed triples.

    ===========  ===========================================
    Pattern      Equation
    ===========  ===========================================
    Composition  $r'(x, y) \land r''(y, z) \implies r(x, z)$
    ===========  ===========================================

    :param mapped_triples_list:
        A collection of ID-based triples.
    :param pairs:
        A mapping from relations to the set of entity pairs.

    :yields:
        A pattern match tuple of relation_id, pattern_type, support, and confidence.
    """
    logger.debug("Evaluating ternary patterns: {composition}")
    # composition r1(x, y) & r2(y, z) => r(x, z)
    # indexing triples for fast join r1 & r2
    adj: DefaultDict[int, DefaultDict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    for h, r, t in mapped_triples_list:
        adj[r][h].add(t)
    # actual evaluation of the pattern
    for r1, r2 in tqdm(_composition_candidates(mapped_triples_list)):
        ht1 = pairs[r1]
        zs = adj[r2]
        lhs = {
            (x, z)
            for (x, y) in ht1
            for z in zs[y]
        }
        support = len(lhs)
        # skip empty support
        # TODO: Can this happen after pre-filtering?
        if not support:
            continue
        for r, ht in pairs.items():
            confidence = len(lhs.intersection(ht)) / support
            yield PatternMatch(r, "composition", support, confidence)


def _determine_patterns(
    mapped_triples_list: Collection[Tuple[int, int, int]],
) -> Iterable[PatternMatch]:
    # indexing triples for fast lookup of entity pair sets
    pairs: DefaultDict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for h, r, t in mapped_triples_list:
        pairs[r].add((h, t))
    # unary
    yield from yield_unary_patterns(pairs=pairs)

    # binary
    yield from yield_binary_patterns(pairs=pairs)

    # ternary
    yield from yield_ternary_patterns(mapped_triples_list, pairs)


def relation_classification(
    dataset: Dataset,
    min_support: int = 0,
    min_confidence: float = 0.95,
    drop_confidence: bool = True,
    parts: Optional[Collection[str]] = None,
) -> pandas.DataFrame:
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

    .. warning ::
        If you intend to use the relation categorization as input to your model, or hyperparameter selection, do *not*
        include testing triples to avoid leakage!

    :return:
        A dataframe with columns {"relation_id", "pattern", "support"?, "confidence"?}.
    """
    # normalize parts
    if parts is None:
        parts = dataset.factory_dict.keys()
    parts = [parts] if isinstance(parts, str) else parts

    # include part hash into cache-file name
    part_hash = hashlib.sha512("".join(sorted(parts)).encode("utf8"))[:16]
    cache_path = PYKEEN_DATASETS.joinpath(dataset.__class__.__name__.lower(), f"relation_patterns_{part_hash}.tsv.xz")

    # re-use cached file if possible
    if not cache_path.is_file():
        # select triples
        mapped_triples = torch.cat([
            dataset.factory_dict[part].mapped_triples
            for part in parts
        ], dim=0)

        # determine patterns from triples
        base = _determine_patterns(mapped_triples_list=mapped_triples.tolist())

        # drop zero-confidence
        base = (
            pattern
            for pattern in base
            if pattern.confidence > 0
        )

        # keep only skyline
        base = skyline(base)

        # create data frame
        df = pandas.DataFrame(
            data=list(base),
            columns=["relation_id", "pattern", "support", "confidence"],
        ).sort_values(by=["pattern", "relation_id", "confidence", "support"])

        # save to file
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(cache_path, sep="\t", index=False)
        logger.info(f"Cached {len(df)} relational pattern entries to {cache_path.as_uri()}")
    else:
        df = pandas.read_csv(cache_path, sep="\t")
        logger.info(f"Loaded {len(df)} precomputed relational patterns from {cache_path.as_uri()}")

    # Prune by support and confidence
    df = df[(df["support"] >= min_support) & (df["confidence"] >= min_confidence)]

    if drop_confidence:
        df = df[["relation_id", "pattern"]].drop_duplicates()

    return df
