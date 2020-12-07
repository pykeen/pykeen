# -*- coding: utf-8 -*-

"""Generate a fake literals dataset to accompany a real dataset.."""

from typing import Any, List, TextIO, Tuple, Type, Union

import click
import torch

import pykeen.nn
from pykeen.datasets.base import Dataset
from pykeen.models.base import EntityRelationEmbeddingModel
from pykeen.pipeline import PipelineResult, pipeline


@click.command()
@click.option('--dataset', required=True)
@click.option('--features', type=int, default=2, show_default=True)
@click.option('--seed', type=int, default=2, show_default=True)
@click.option('--output', type=click.File('w'))
def main(dataset, features: int, seed: int, output: TextIO):
    """Generate random literals for the Nations dataset."""
    for h, r, t in generate_literals(dataset=dataset, features=features, seed=seed):
        print(h, r, t, sep='\t', file=output)


def generate_literals(
    dataset: Union[None, str, Dataset, Type[Dataset]],
    features: int,
    seed: int,
    model: str = 'rotate',
) -> List[Tuple[str, str, Any]]:
    """Generate literals or the given dataset."""
    generator = torch.Generator(seed)

    pipeline_result: PipelineResult = pipeline(
        dataset=dataset,
        model=model,
        training_kwargs=dict(
            num_epochs=120,
        ),
        random_seed=seed,
    )

    assert isinstance(pipeline_result.model, EntityRelationEmbeddingModel)

    tf = pipeline_result.training_loop.triples_factory
    # calculate a numeric value for each nation based on embeddings to make a high correlation
    entity_embeddings: pykeen.nn.Embedding = pipeline_result.model.entity_embeddings

    rows = []
    for i in range(features):
        feature_matrix = torch.rand(size=(entity_embeddings.embedding_dim,), generator=generator)
        feature = entity_embeddings.forward(None) @ feature_matrix
        # Add some noise
        feature += torch.normal(mean=0, std=1, size=feature.size(), generator=generator)
        for (_, label), value in zip(sorted(tf.entity_id_to_label.items()), feature):
            rows.append((label, f'feature{i}', value.item()))
    return rows


if __name__ == '__main__':
    main()
