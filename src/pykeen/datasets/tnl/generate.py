# -*- coding: utf-8 -*-

"""Generate the Fake Nations Literal Dataset."""

import click
import torch

import pykeen.nn
from pykeen.datasets.tnl import LITERALS_PATH
from pykeen.models.base import EntityRelationEmbeddingModel
from pykeen.pipeline import PipelineResult, pipeline


@click.command()
@click.option('--features', type=int, default=2, show_default=True)
@click.option('--seed', type=int, default=2, show_default=True)
def generate(features: int, seed: int):
    """Generate random literals for the Nations dataset."""
    generator = torch.Generator(seed)

    pipeline_result: PipelineResult = pipeline(
        dataset='nations',
        model='rotate',
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

    with open(LITERALS_PATH, 'w') as file:
        for h, r, t in rows:
            print(h, r, t, sep='\t', file=file)
