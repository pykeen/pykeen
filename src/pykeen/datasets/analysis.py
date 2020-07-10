# -*- coding: utf-8 -*-

"""Dataset analysis utilities."""

from operator import itemgetter

import numpy
import pandas
import torch

from .base import DataSet
from ..utils import invert_mapping

SUBSET_LABELS = ('testing', 'training', 'validation', 'total')


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


def relation_count_dataframe(dataset: DataSet) -> pandas.DataFrame:
    """Create a dataframe with relation counts for all subsets, and the full dataset.

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


def entity_count_dataframe(dataset: DataSet) -> pandas.DataFrame:
    """Create a dataframe with head/tail/both counts for all subsets, and the full dataset.

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
    df = pandas.DataFrame(data=data, index=index, columns=pandas.MultiIndex.from_product(iterables=[SUBSET_LABELS, second_level_order]))
    df.index.name = 'entity_label'
    return df


def entity_relation_co_occurrence_dataframe(dataset: DataSet) -> pandas.DataFrame:
    """Create a dataframe of entity/relation co-occurrence.

    This information can be seen as a form of pseudo-typing, e.g. entity A is something which can be a head of
    `born_in`.

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
