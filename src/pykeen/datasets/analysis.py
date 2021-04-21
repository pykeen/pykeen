# -*- coding: utf-8 -*-

"""Dataset analysis utilities."""

import logging
from typing import Collection, Optional, Tuple, Union

import numpy
import pandas as pd
import torch

from .base import Dataset
from ..constants import PYKEEN_DATASETS
from ..triples.analysis import get_id_counts, iter_relation_cardinality_types, relation_classification_tmp, triple_set_hash
from ..utils import invert_mapping

logger = logging.getLogger(__name__)

__all__ = [
    'relation_classification',
    'relation_count_dataframe',
    'entity_count_dataframe',
    'entity_relation_co_occurrence_dataframe',
    'calculate_relation_functionality',
]

SUBSET_LABELS = ('testing', 'training', 'validation', 'total')


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

        df = relation_classification_tmp(mapped_triples)

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


def _add_relation_labels(
    dataset: Dataset,
    df: pd.DataFrame,
    relation_id_column: str = "relation_id",
    relation_label_column: str = "relation_label",
) -> pd.DataFrame:
    if dataset.relation_to_id is None:
        return df
    return pd.merge(
        left=df,
        right=pd.DataFrame(
            data=list(dataset.relation_to_id.items()),
            columns=[relation_label_column, relation_id_column],
        ),
        on=relation_id_column,
    )


def relation_cardinality_classification(
    *,
    dataset: Dataset,
    parts: Optional[Collection[str]] = None,
    add_labels: bool = True,
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

    :param dataset:
        The dataset to investigate.
    :param parts:
        Only use certain parts of the dataset, e.g., train triples. Defaults to using all triples, i.e.
        {"training", "validation", "testing}.
    :param add_labels:
        Whether to add relation labels (if available).

    :return:
        A dataframe with columns ( relation_id | relation_type )
    """
    # TODO: Consider merging with other analysis methods
    parts = _normalize_parts(dataset=dataset, parts=parts)
    mapped_triples = _get_mapped_triples(dataset=dataset, parts=parts)

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
        columns=["relation_id", "relation_type", "support", "confidence"],
    )
    if add_labels:
        df = _add_relation_labels(dataset=dataset, df=df)
    return df


def calculate_relation_functionality(
    *,
    dataset: Dataset,
    parts: Optional[Collection[str]] = None,
    add_labels: bool = True,
) -> pd.DataFrame:
    """
    Calculate the functionality and inverse functionality score per relation.

    The (inverse) functionality was proposed in [wang2018]_. It is defined as the number of unique head (tail) entities
    divided by the of triples in which the relation occurs. Thus, its value range is [0, 1]. Smaller values indicate
    that entities usually have more than one outgoing (incoming) triple with the corresponding relation type. Hence,
    the score is related to the relation cardinality types.

    :param dataset:
        The dataset to investigate.
    :param parts:
        Only use certain parts of the dataset, e.g., train triples. Defaults to using all triples, i.e.
        {"training", "validation", "testing}.
    :param add_labels:
        Whether to add relation labels (if available).

    :return:
        A dataframe with columns (relation_id | functionality | inverse functionality)

    .. [wang2018]
        Wang, Z., *et al.* (2018). `Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks
        <https://doi.org/10.18653/v1/D18-1032>`_. Proceedings of the 2018 Conference on Empirical Methods in
        Natural Language Processing, 349–357.
    """
    # TODO: Consider merging with other analysis methods
    parts = _normalize_parts(dataset=dataset, parts=parts)
    mapped_triples = _get_mapped_triples(dataset=dataset, parts=parts)
    df = pd.DataFrame(data=mapped_triples, columns=["h", "r", "t"])
    df = df.groupby(by="r").agg(dict(
        h=["nunique", "count"],
        t="nunique",
    ))
    df["functionality"] = df[("h", "nunique")] / df[("h", "count")]
    df["inverse_functionality"] = df[("t", "nunique")] / df[("h", "count")]
    df = df[["functionality", "inverse_functionality"]]
    df.columns = df.columns.droplevel(1)
    df.index.name = "relation_id"
    df = df.reset_index()

    if add_labels:
        df = _add_relation_labels(dataset=dataset, df=df)
    return df
