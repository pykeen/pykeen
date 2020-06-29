from typing import Any, Mapping

import pandas
import torch

from pykeen.datasets import DataSet
from pykeen.typing import MappedTriples


def describe_column(
    column: torch.LongTensor,
    max_id: int,
):
    unique, counts = column.unique(return_counts=True)
    series = pandas.Series(counts)
    return dict(
        num_missing_ids=max_id - len(unique),
        **series.describe().to_dict(),
    )


def describe_triples(
    mapped_triples: MappedTriples,
    num_entities: int,
    num_relations: int,
) -> Mapping[str, Any]:
    return dict(
        head=describe_column(column=mapped_triples[:, 0], max_id=num_entities),
        tail=describe_column(column=mapped_triples[:, 1], max_id=num_relations),
        relation=describe_column(column=mapped_triples[:, 2], max_id=num_entities),
        entity=describe_column(column=torch.cat([mapped_triples[:, col] for col in (0, 2)], dim=0), max_id=num_entities),
    )


def describe_dataset(
    dataset: DataSet,
) -> Mapping[str, Any]:
    """Describe a dataset by computing numerous statistics.

    :param dataset:
        The dataset.

    :return:
        A JSON-compatible description.
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
            )
            for subset_name, triples_factory in dataset.factory_dict.items()
        },
        description=describe_triples(
            mapped_triples=torch.cat([factory.mapped_triples for factory in dataset.factories], dim=0),
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
        )
    )
