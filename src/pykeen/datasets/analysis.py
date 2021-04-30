# -*- coding: utf-8 -*-

"""Dataset analysis utilities."""

import functools
import logging
from typing import Collection, Mapping, Optional, Tuple, Union

import numpy
import pandas as pd
import torch

from .base import Dataset
from ..constants import PYKEEN_DATASETS
from ..triples import CoreTriplesFactory, analysis as triple_analysis
from ..triples.analysis import _get_counts
from ..typing import MappedTriples
from ..utils import invert_mapping

logger = logging.getLogger(__name__)

__all__ = [
    "get_relation_count_df",
    "get_entity_count_df",
    "get_entity_relation_co_occurrence_df",
    "get_relation_functionality_df",
    # relation typing
    "get_relation_pattern_types_df",
    "get_relation_cardinality_types_df"
]

# constants
COUNT_COLUMN_NAME = "count"
ENTITY_ID_COLUMN_NAME = "entity_id"
ENTITY_LABEL_COLUMN_NAME = "entity_label"
RELATION_ID_COLUMN_NAME = "relation_id"
RELATION_LABEL_COLUMN_NAME = "relation_label"
SUBSET_COLUMN_NAME = "subset"

#: fixme: deprecated
SUBSET_LABELS = ("testing", "training", "validation", "total")


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
    # unique
    return list(set(parts))


def _add_labels(
    label_to_id: Optional[Mapping[str, int]],
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
) -> pd.DataFrame:
    if label_to_id is None:
        return df
    return pd.merge(
        left=df,
        right=pd.DataFrame(
            data=list(label_to_id.items()),
            columns=[label_column, id_column],
        ),
        on=id_column,
    )


_add_entity_labels = functools.partial(
    _add_labels,
    id_column=ENTITY_ID_COLUMN_NAME,
    label_column=ENTITY_LABEL_COLUMN_NAME,
)

_add_relation_labels = functools.partial(
    _add_labels,
    id_column=RELATION_ID_COLUMN_NAME,
    label_column=RELATION_LABEL_COLUMN_NAME,
)


def get_relation_count_df(
    *,
    dataset: Optional[Dataset] = None,
    triples_factory: Optional[CoreTriplesFactory] = None,
    mapped_triples: Optional[MappedTriples] = None,
    parts: Optional[Collection[str]] = None,
    add_labels: bool = True,
    total_count: bool = False,
    relation_to_id: Mapping[str, int] = None,
) -> pd.DataFrame:
    """Create a dataframe with relation counts.

    Example usage:

    >>> from pykeen.datasets import Nations
    >>> dataset = Nations()
    >>> from pykeen.datasets.analysis import get_relation_count_df
    >>> df = get_relation_count_df(dataset=dataset)

    # Get the most frequent relations in training
    >>> df[df["subset"] == "training"].sort_values(by="count", ascending=False).head()

    # Get all relations which do not occur in the test part
    >>> df[(df["subset"] == "testing") & (df["count"] == 0)]

    :param dataset:
        The dataset.
    :param triples_factory:
        The triples factory.
    :param mapped_triples:
        The mapped triples.
    :param parts:
        Can be used in conjuction with dataset to select only a part of the triples factories.
    :param add_labels:
        Whether to add relation labels to the dataframe. Requires the triples factory / dataset to provide labels, or
        to explicitly provide a relation_to_id mapping.
    :param total_count:
        Whether to combine the counts from all subsets, or keep the separate counts instead. Only effective if a dataset
        is given.
    :param relation_to_id:
        A relation label to Id mapping. Takes precedence over mappings from triples_factory / dataset.

    :return:
        A dataframe with columns (relation_id, count, relation_label?, subset?)
    """
    if sum(x is not None for x in (mapped_triples, triples_factory, dataset)) != 1:
        raise ValueError("Exactly one of {mapped_triples, triples_factory, dataset} must be not None.")

    if mapped_triples is not None:
        df = pd.DataFrame(data=dict(zip(
            [RELATION_ID_COLUMN_NAME, COUNT_COLUMN_NAME],
            _get_counts(mapped_triples=mapped_triples, column=1),
        )))
    elif triples_factory is not None:
        df = get_relation_count_df(
            mapped_triples=triples_factory.mapped_triples,
            add_labels=False,  # are added after aggregation
        )
    else:
        parts = _normalize_parts(dataset=dataset, parts=parts)
        data = []
        for subset_name in parts:
            df = get_relation_count_df(
                triples_factory=dataset.factory_dict[subset_name],
                add_labels=False,  # are added after aggregation
            )
            if not total_count:
                df["subset"] = subset_name
            data.append(df)
        df = pd.concat(data, ignore_index=True)
        if total_count:
            df = df.groupby(by=RELATION_ID_COLUMN_NAME)[COUNT_COLUMN_NAME].sum().reset_index()

    if not add_labels:
        return df

    # Infer relation to id mapping
    if not relation_to_id and triples_factory and hasattr(triples_factory, "relation_to_id"):
        relation_to_id = triples_factory.relation_to_id
    if not relation_to_id and dataset and hasattr(dataset, "relation_to_id"):
        relation_to_id = dataset.relation_to_id
    if relation_to_id is None:
        raise ValueError(
            "To add relation labels, either an relation_to_id mapping has to be explicitly provided, or the "
            "triples_factory / dataset must provide such mapping."
        )

    # add label column
    return _add_relation_labels(df=df, label_to_id=relation_to_id)


def get_entity_count_df(
    dataset: Dataset,
    both_sides: bool = True,
    total_count: bool = True,
    add_labels: bool = True,
) -> pd.DataFrame:
    """Create a dataframe with head/tail/both counts for all subsets, and the full dataset.

    Example usage:

    >>> from pykeen.datasets import FB15k237
    >>> dataset = FB15k237()
    >>> from pykeen.datasets.analysis import get_relation_count_df
    >>> df = get_entity_count_df(dataset=dataset)

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
    # TODO: Merge duplicate code
    data = []
    for subset_name, triples_factory in dataset.factory_dict.items():
        df = triple_analysis.get_entity_counts(mapped_triples=triples_factory.mapped_triples)
        df["subset"] = subset_name
        data.append(df)
    df = pd.concat(data, ignore_index=True)
    if both_sides:
        df = df.groupby([ENTITY_ID_COLUMN_NAME, "subset"])["count"].sum().reset_index()
    if total_count:
        group_key = [ENTITY_ID_COLUMN_NAME]
        if not both_sides:
            group_key += ["type"]
        df = df.groupby(by=group_key)["count"].sum().reset_index()
    if add_labels:
        df = _add_entity_labels(label_to_id=dataset.entity_to_id, df=df)
    return df


def get_entity_relation_co_occurrence_df(dataset: Dataset) -> pd.DataFrame:
    """Create a dataframe of entity/relation co-occurrence.

    This information can be seen as a form of pseudo-typing, e.g. entity A is something which can be a head of
    `born_in`.

    Example usages:
    >>> from pykeen.datasets import Nations
    >>> dataset = Nations()
    >>> from pykeen.datasets.analysis import get_relation_count_df
    >>> df = get_entity_count_df(dataset=dataset)

    # Which countries have to most embassies (considering only training triples)?
    >>> df.loc["training", ("head", "embassy")].sort_values().tail()

    # In which countries are to most embassies (considering only training triples)?
    >>> df.loc["training", ("tail", "embassy")].sort_values().tail()

    :param dataset:
        The dataset.

    :return:
        A dataframe with a multi-index (subset, entity_id) as index, and a multi-index (kind, relation) as columns,
        where subset in {"training", "validation", "testing", "total"}, and kind in {"head", "tail"}. For each entity,
        the corresponding row can be seen a pseudo-type, i.e. for which relations it may occur as head/tail.
    """
    # TODO: Update to long form
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
            sorted(dataset.factory_dict.keys()) + ["total"],
            [entity_id_to_label[entity_id] for entity_id in range(num_entities)],
        ]),
        columns=pd.MultiIndex.from_product([
            ("head", "tail"),
            [relation_id_to_label[relation_id] for relation_id in range(num_relations)],
        ]),
    )


def get_relation_pattern_types_df(
    dataset: Dataset,
    *,
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
    ph = triple_analysis.triple_set_hash(mapped_triples=mapped_triples)[:16]

    # include part hash into cache-file name
    cache_path = PYKEEN_DATASETS.joinpath(dataset.__class__.__name__.lower(), f"relation_patterns_{ph}.tsv.xz")

    # re-use cached file if possible
    if not cache_path.is_file() or force:
        # select triples
        mapped_triples = torch.cat([
            dataset.factory_dict[part].mapped_triples
            for part in parts
        ], dim=0).tolist()

        df = triple_analysis.relation_pattern_types(mapped_triples=mapped_triples)

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

    df = triple_analysis.relation_cardinality_types(mapped_triples=mapped_triples)
    if add_labels:
        df = _add_relation_labels(df=df, label_to_id=dataset.relation_to_id)
    return df


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
        df = _add_relation_labels(df=df, label_to_id=dataset.relation_to_id)
    return df
