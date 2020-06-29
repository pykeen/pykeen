from typing import Any, Mapping

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
