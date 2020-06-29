from typing import Any, Mapping

import numpy
import pandas
import torch

from pykeen.datasets import DataSet
from pykeen.typing import MappedTriples
from pykeen.utils import invert_mapping


def describe_id_tensor(
    tensor: torch.LongTensor,
    max_id: int,
    label_to_id: Mapping[str, int],
    k: int = 10,
) -> Mapping[str, Any]:
    """Describe a tensor of IDs.

    In particular computes

    :param tensor: shape: (num,)
        The tensor of IDs.
    :param max_id:
        The maximum ID. Valid IDs are in {0, 1, ..., max_id - 1}.
    :param label_to_id:
        A mapping from labels to IDs.
    :param k:
        The number of most frequent IDs to compute.

    :return:
        A JSON-compatible description of the ID-tensor.
    """
    # Calculate ID frequencies
    unique, counts = tensor.unique(return_counts=True)

    # descriptive statistics via pandas
    frequency = pandas.Series(counts).describe().to_dict()

    # Get top-k IDs
    top_counts, top_ids = counts.topk(k=k, largest=True, sorted=True)
    top_ids = unique[top_ids].tolist()
    id_to_label = invert_mapping(mapping=label_to_id)
    top = [(id_to_label[id_], count) for id_, count in zip(top_ids, top_counts.tolist())]

    return dict(
        num_missing_ids=max_id - len(unique),
        top=top,
        frequency=frequency,
    )


def describe_triples(
    mapped_triples: MappedTriples,
    num_entities: int,
    num_relations: int,
    entity_to_id: Mapping[str, int],
    relation_to_id: Mapping[str, int],
) -> Mapping[str, Any]:
    """Describe a tensor of triples.

    :param mapped_triples:
        The mapped triples (i.e. ID-based).
    :param num_entities:
        The number of entities.
    :param num_relations:
        The number of relations.
    :param entity_to_id:
        A mapping of entity label to ID.
    :param relation_to_id:
        A mapping of relation label to ID.

    :return:
        A JSON-compatible description of the triples. Comprises descriptions of each individual column
        (head, relation, tail), as well as of all entity IDs (i.e. head and tail)
    """
    return dict(
        head=describe_id_tensor(tensor=mapped_triples[:, 0], max_id=num_entities, label_to_id=entity_to_id),
        relation=describe_id_tensor(tensor=mapped_triples[:, 1], max_id=num_relations, label_to_id=relation_to_id),
        tail=describe_id_tensor(tensor=mapped_triples[:, 2], max_id=num_entities, label_to_id=entity_to_id),
        entity=describe_id_tensor(
            tensor=torch.cat([mapped_triples[:, col] for col in (0, 2)], dim=0),
            max_id=num_entities,
            label_to_id=entity_to_id,
        ),
    )


def describe_dataset(
    dataset: DataSet,
) -> Mapping[str, Any]:
    """Describe a dataset by computing numerous statistics.

    :param dataset:
        The dataset.

    :return:
        A JSON-compatible description. Comprises descriptions of the each subset, train/test/validation, as well as the
         union over the subsets.
    """
    return dict(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        total_num_triples=sum(factory.num_triples for factory in dataset.factories),
        subset_descriptions={
            subset_name: describe_triples(
                mapped_triples=triples_factory.mapped_triples,
                num_entities=dataset.num_entities,
                num_relations=dataset.num_relations,
                entity_to_id=dataset.entity_to_id,
                relation_to_id=dataset.relation_to_id,
            )
            for subset_name, triples_factory in dataset.factory_dict.items()
        },
        description=describe_triples(
            mapped_triples=torch.cat([factory.mapped_triples for factory in dataset.factories], dim=0),
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            entity_to_id=dataset.entity_to_id,
            relation_to_id=dataset.relation_to_id,
        )
    )


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


def entity_count_dataframe(
    dataset: DataSet,
) -> pandas.DataFrame:
    """Create a dataframe with head/tail/both counts for all subsets, and the full dataset.

    :param dataset:
        The dataset.

    :return:
        A dataframe with one row per entity.
    """
    data = {}
    num_entities = dataset.num_entities
    for subset_name, triples_factory in dataset.factory_dict.items():
        for col, col_name in zip((0, 2), ('head', 'tail')):
            data[(subset_name, col_name)] = get_id_counts(
                id_tensor=triples_factory.mapped_triples[:, col],
                num_ids=num_entities,
            )
        data[(subset_name, 'total')] = data[(subset_name, 'head')] + data[(subset_name, 'tail')]
    for kind in ('head', 'tail', 'total'):
        data[('total', kind)] = sum(data[(subset_name, kind)] for subset_name in dataset.factory_dict.keys())
    return pandas.DataFrame(
        data=data,
        index=sorted(dataset.entity_to_id.items(), key=lambda label_id: label_id[1]),
    )


def entity_relation_co_occurrence_dataframe(
    dataset: DataSet,
) -> pandas.DataFrame:
    """Create a dataframe of entity/relation co-occurence.

    :param dataset:
        The dataset.

    :return:
        A dataframe with a multi-index (subset, entity_id) as index, and a multi-index (kind, relation) as columns, where
        subset in {'training', 'validation', 'testing', 'total'}, and kind in {'head', 'tail'}. For each entity, the
        corresponding row can be seen a pseudo-type, i.e. for which relations it may occur as head/tail.
    """
    num_relations = dataset.num_relations
    num_entities = dataset.num_entities
    data = numpy.zeros(shape=(4 * num_entities, 2 * num_relations), dtype=numpy.int64)
    for i, (subset_name, triples_factory) in enumerate(sorted(dataset.factory_dict.items())):
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
