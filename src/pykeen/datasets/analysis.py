# -*- coding: utf-8 -*-

"""Dataset analysis utilities."""

import logging
from typing import Callable, Collection, Optional, Tuple, Union

import pandas as pd
import torch

from .base import Dataset
from ..constants import PYKEEN_DATASETS
from ..triples import analysis as triple_analysis
from ..typing import MappedTriples

logger = logging.getLogger(__name__)

__all__ = [
    "get_relation_count_df",
    "get_entity_count_df",
    "get_entity_relation_co_occurrence_df",
    "get_relation_functionality_df",
    # relation typing
    "get_relation_pattern_types_df",
    "get_relation_cardinality_types_df",
]

# constants
SUBSET_COLUMN_NAME = "subset"


def _get_mapped_triples(dataset: Dataset, parts: Collection[str]) -> Collection[Tuple[int, int, int]]:
    return torch.cat([dataset.factory_dict[part].mapped_triples for part in parts], dim=0).tolist()


def _normalize_parts(dataset: Dataset, parts: Union[None, str, Collection[str]]) -> Collection[str]:
    if parts is None:
        parts = dataset.factory_dict.keys()
    elif isinstance(parts, str):
        parts = [parts]
    # unique
    return list(set(parts))


def _common(
    dataset: Dataset,
    triple_func: Callable[[MappedTriples], pd.DataFrame],
    merge_sides: bool = True,
    merge_subsets: bool = True,
    add_labels: bool = True,
) -> pd.DataFrame:
    """
    Execute triple analysis over a dataset.

    :param dataset:
        The dataset.
    :param triple_func:
        The analysis function on the triples.
    :param merge_sides:
        Whether to merge sides, i.e., entity positions: head vs. tail.
    :param merge_subsets:
        Whether to merge subsets, i.e., train/validation/test.
    :param add_labels:
        Whether to add entity / relation labels.

    :return:
        An aggregated dataframe.
    """
    # compute over all triples
    data = []
    for subset_name, triples_factory in dataset.factory_dict.items():
        df = triple_func(triples_factory.mapped_triples)
        df[SUBSET_COLUMN_NAME] = subset_name
        data.append(df)
    df = pd.concat(data, ignore_index=True)

    # Determine group key
    group_key = []
    for key, condition in (
        (triple_analysis.ENTITY_ID_COLUMN_NAME, True),
        (triple_analysis.RELATION_ID_COLUMN_NAME, True),
        (triple_analysis.ENTITY_POSITION_COLUMN_NAME, not merge_sides),
        (SUBSET_COLUMN_NAME, not merge_subsets),
    ):
        if condition and key in df.columns:
            group_key.append(key)
    df = df.groupby(by=group_key)[triple_analysis.COUNT_COLUMN_NAME].sum().reset_index()

    # Add labels if requested
    if add_labels and triple_analysis.ENTITY_ID_COLUMN_NAME in df.columns:
        df = triple_analysis.add_entity_labels(
            df=df,
            add_labels=add_labels,
            label_to_id=dataset.entity_to_id,
        )
    if add_labels and triple_analysis.RELATION_ID_COLUMN_NAME in df.columns:
        df = triple_analysis.add_relation_labels(
            df=df,
            add_labels=add_labels,
            label_to_id=dataset.relation_to_id,
        )
    return df


def get_relation_count_df(
    dataset: Dataset,
    merge_subsets: bool = True,
    add_labels: bool = True,
) -> pd.DataFrame:
    """Create a dataframe with relation counts.

    :param dataset:
        The dataset.
    :param add_labels:
        Whether to add relation labels to the dataframe.
    :param merge_subsets:
        Whether to merge subsets, i.e., train/validation/test.
    :param add_labels:
        Whether to add entity / relation labels.

    :return:
        A dataframe with columns (relation_id, count, relation_label?, subset?)
    """
    return _common(
        dataset=dataset,
        triple_func=triple_analysis.get_relation_counts,
        merge_subsets=merge_subsets,
        add_labels=add_labels,
    )


def get_entity_count_df(
    dataset: Dataset,
    merge_sides: bool = True,
    merge_subsets: bool = True,
    add_labels: bool = True,
) -> pd.DataFrame:
    """Create a dataframe with entity counts.

    :param dataset:
        The dataset.
    :param merge_sides:
        Whether to merge sides, i.e., entity positions: head vs. tail.
    :param merge_subsets:
        Whether to merge subsets, i.e., train/validation/test.
    :param add_labels:
        Whether to add entity / relation labels.

    :return:
        A dataframe with one row per entity.
    """
    return _common(
        dataset=dataset,
        triple_func=triple_analysis.get_entity_counts,
        merge_sides=merge_sides,
        merge_subsets=merge_subsets,
        add_labels=add_labels,
    )


def get_entity_relation_co_occurrence_df(
    dataset: Dataset,
    merge_sides: bool = True,
    merge_subsets: bool = True,
    add_labels: bool = True,
) -> pd.DataFrame:
    """Create a dataframe of entity/relation co-occurrence.

    This information can be seen as a form of pseudo-typing, e.g. entity A is something which can be a head of
    `born_in`.

    :param dataset:
        The dataset.
    :param merge_sides:
        Whether to merge sides, i.e., entity positions: head vs. tail.
    :param merge_subsets:
        Whether to merge subsets, i.e., train/validation/test.
    :param add_labels:
        Whether to add entity / relation labels.

    :return:
        A dataframe of entity-relation pairs with their occurrence count.
    """
    return _common(
        dataset=dataset,
        triple_func=triple_analysis.entity_relation_co_occurrence,
        merge_sides=merge_sides,
        merge_subsets=merge_subsets,
        add_labels=add_labels,
    )


def get_relation_pattern_types_df(
    dataset: Dataset,
    *,
    min_support: int = 0,
    min_confidence: float = 0.95,
    drop_confidence: bool = False,
    parts: Optional[Collection[str]] = None,
    force: bool = False,
    add_labels: bool = True,
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
    :param add_labels:
        Whether to add relation labels (if available).

    .. warning ::

        If you intend to use the relation categorization as input to your model, or hyper-parameter selection, do *not*
        include testing triples to avoid leakage!

    :return:
        A dataframe with columns {"relation_id", "pattern", "support"?, "confidence"?}.
    """
    # TODO: Merge with _common?
    parts = _normalize_parts(dataset, parts)
    mapped_triples = _get_mapped_triples(dataset, parts)

    # include hash over triples into cache-file name
    ph = triple_analysis.triple_set_hash(mapped_triples=mapped_triples)[:16]

    # include part hash into cache-file name
    cache_path = PYKEEN_DATASETS.joinpath(dataset.__class__.__name__.lower(), f"relation_patterns_{ph}.tsv.xz")

    # re-use cached file if possible
    if not cache_path.is_file() or force:
        # select triples
        mapped_triples = torch.cat([dataset.factory_dict[part].mapped_triples for part in parts], dim=0).tolist()

        df = triple_analysis.relation_pattern_types(mapped_triples=mapped_triples)

        # save to file
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(cache_path, sep="\t", index=False)
        logger.info(f"Cached {len(df)} relational pattern entries to {cache_path.as_uri()}")
    else:
        df = pd.read_csv(cache_path, sep="\t")
        logger.info(f"Loaded {len(df)} precomputed relational patterns from {cache_path.as_uri()}")

    # Prune by support and confidence
    sufficient_support = df[triple_analysis.SUPPORT_COLUMN_NAME] >= min_support
    sufficient_confidence = df[triple_analysis.CONFIDENCE_COLUMN_NAME] >= min_confidence
    df = df[sufficient_support & sufficient_confidence]

    if drop_confidence:
        df = df[[triple_analysis.RELATION_ID_COLUMN_NAME, triple_analysis.PATTERN_TYPE_COLUMN_NAME]].drop_duplicates()

    return triple_analysis.add_relation_labels(
        df=df,
        add_labels=add_labels,
        label_to_id=dataset.relation_to_id,
    )


def get_relation_cardinality_types_df(
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
    return triple_analysis.relation_cardinality_types(
        mapped_triples=mapped_triples,
        add_labels=add_labels,
        label_to_id=dataset.relation_to_id,
    )


def get_relation_injectivity_df(
    *,
    dataset: Dataset,
    parts: Optional[Collection[str]] = None,
    add_labels: bool = True,
) -> pd.DataFrame:
    """
    Calculate "soft" injectivity scores for each relation.

    :param dataset:
        The dataset to investigate.
    :param parts:
        Only use certain parts of the dataset, e.g., train triples. Defaults to using all triples, i.e.
        {"training", "validation", "testing}.
    :param add_labels:
        Whether to add relation labels (if available).

    :return:
        A dataframe with one row per relation, its number of occurrences and head / tail injectivity scores.
    """
    # TODO: Consider merging with other analysis methods
    parts = _normalize_parts(dataset=dataset, parts=parts)
    mapped_triples = _get_mapped_triples(dataset=dataset, parts=parts)
    return triple_analysis.relation_injectivity(
        mapped_triples=mapped_triples,
        add_labels=add_labels,
        label_to_id=dataset.relation_to_id,
    )


def get_relation_functionality_df(
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
    return triple_analysis.get_relation_functionality(
        mapped_triples,
        add_labels=add_labels,
        label_to_id=dataset.relation_to_id,
    )
