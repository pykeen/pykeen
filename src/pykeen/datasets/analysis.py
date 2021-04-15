# -*- coding: utf-8 -*-

"""Dataset analysis utilities."""
import itertools
import logging
from operator import itemgetter
from typing import Iterable, Tuple

import numpy
import pandas
import torch

from .base import Dataset
from ..utils import invert_mapping

SUBSET_LABELS = ('testing', 'training', 'validation', 'total')

logger = logging.getLogger(__name__)


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
            [entity_id_to_label[entity_id] for entity_id in range(num_entities)]
        ]),
        columns=pandas.MultiIndex.from_product([
            ('head', 'tail'),
            [relation_id_to_label[relation_id] for relation_id in range(num_relations)]
        ])
    )


def _relation_classification_pandas(
    triple_df: pandas.DataFrame,
) -> Iterable[Tuple[int, str, int, float]]:
    """
    The actual work behind relation classification, built upon pandas.

    .. note ::
        The implementation is quite memory-intense.
    """
    support = triple_df.value_counts(subset=["r"]).rename("support").reset_index()

    # unary patterns
    logger.info("Checking unary patterns: {symmetry, anti-symmetry}")
    # symmetry: r(x, y) => r(y, x)
    temp_df = pandas.merge(
        left=triple_df,
        right=triple_df,
        left_on=["r", "h", "t"],
        right_on=["r", "t", "h"],
    ).groupby(by="r").size().rename("full_count").reset_index()
    temp_df = pandas.merge(
        left=support,
        right=temp_df,
        on="r",
    )
    temp_df["confidence"] = temp_df["full_count"] / temp_df["support"]
    yield from zip(
        temp_df["r"].tolist(),
        itertools.repeat("symmetry"),
        temp_df["support"].tolist(),
        temp_df["confidence"].tolist(),
    )

    # anti-symmetry: r(x, y) => !r(y, x)
    temp_df["confidence"] = (temp_df["support"] - temp_df["full_count"]) / temp_df["support"]
    yield from zip(
        temp_df["r"].tolist(),
        itertools.repeat("anti-symmetry"),
        temp_df["support"].tolist(),
        temp_df["confidence"].tolist(),
    )

    # binary patterns
    logger.info("Checking binary patterns: {inversion}")
    # inversion: r1(x, y) => r(x, y)
    temp_df = pandas.merge(
        left=triple_df.rename(columns=dict(r="r1")),
        right=triple_df,
        left_on=["h", "t"],
        right_on=["t", "h"],
    ).groupby(by=["r", "r1"]).size().rename("full_count").reset_index()
    temp_df = pandas.merge(
        left=support,
        right=temp_df,
        on="r",
    )
    temp_df["confidence"] = (temp_df["support"] - temp_df["full_count"]) / temp_df["support"]
    yield from zip(
        temp_df["r"].tolist(),
        itertools.repeat("inversion"),
        temp_df["support"].tolist(),
        temp_df["confidence"].tolist(),
    )

    # ternary patterns
    logger.info("Checking ternary patterns: {composition}")
    # composition r1(x, y) & r2(y, z) => r3(x, z)
    temp_df = pandas.merge(
        left=triple_df.rename(columns=dict(h="x", r="r1", t="y")),
        right=triple_df.rename(columns=dict(h="y", r="r2", t="z")),
        on="y",
    )
    support = temp_df.groupby(by=["r1", "r2"]).size().rename(index="support").reset_index()
    temp_df = pandas.merge(
        left=temp_df,
        right=triple_df.rename(columns=dict(h="x", r="r3", t="z")),
        on=["x", "z"],
    ).groupby(by=["r1", "r2", "r3"]).size().rename(index="full_count").reset_index()
    temp_df = pandas.merge(left=support, right=temp_df, on=["r1", "r2"])
    temp_df["confidence"] = temp_df["full_count"] / temp_df["support"]
    yield from zip(
        temp_df["r3"].tolist(),
        itertools.repeat("composition"),
        temp_df["support"].tolist(),
        temp_df["confidence"].tolist(),
    )


def relation_classification(
    dataset: Dataset,
    min_support: int = 0,
    min_confidence: float = 0.95,
    drop_confidence: bool = True,
) -> pandas.DataFrame:
    r"""
    Compute relation classification based on RotatE [...]_.

    The relation classifications are based upon checking whether the corresponding rules hold with sufficient support
    and confidence. By default, we do not require a minimum support, however, a relatively high confidence.

    The following four non-exclusive classes for relations are considered.

    symmetry:

    .. math ::
        r(x, y) \implies r(y, x)

    anti-symmetry:

    .. math ::
        r(x, y) \implies \neg r(y, x)

    inversion:

    .. math ::
        r'(x, y) \implies r(y, x)

    composition

    .. math ::
        r'(x, y) \land r''(y, z) \implies r(x, z)
    """
    # use all triples; TODO: should we do this?
    mapped_triples = torch.cat([
        triples_factory.mapped_triples
        for triples_factory in dataset.factories
    ], dim=0)

    # convert to dataframe
    triple_df = pandas.DataFrame(data=mapped_triples.numpy(), columns=["h", "r", "t"])

    # Get data
    df = pandas.DataFrame(
        data=[
            (relation_id, pattern, support, confidence)
            for (relation_id, pattern, support, confidence) in _relation_classification_pandas(triple_df=triple_df)
            if support >= min_support and confidence >= min_confidence
        ],
        columns=["relation_id", "pattern", "support", "confidence"],
    ).sort_values(by=["relation_id", "confidence", "support"])

    if drop_confidence:
        df = df[["relation_id", "pattern"]].drop_duplicates()

    return df
