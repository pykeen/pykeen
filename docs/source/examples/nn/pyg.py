"""Example for using PyTorch Geometric."""

from pykeen.datasets import get_dataset
from pykeen.models import ERModel
from pykeen.nn.init import LabelBasedInitializer
from pykeen.pipeline import pipeline

dataset = get_dataset(dataset="nations", dataset_kwargs=dict(create_inverse_triples=True))
entity_initializer = LabelBasedInitializer.from_triples_factory(
    triples_factory=dataset.training,
    for_entities=True,
)
(embedding_dim,) = entity_initializer.tensor.shape[1:]
r = pipeline(
    dataset=dataset,
    model=ERModel,
    model_kwargs=dict(
        interaction="distmult",
        entity_representations="SimpleMessagePassing",
        entity_representations_kwargs=dict(
            triples_factory=dataset.training,
            base_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                trainable=False,
            ),
            layers=["GCN"] * 2,
            layers_kwargs=dict(in_channels=embedding_dim, out_channels=embedding_dim),
        ),
        relation_representations_kwargs=dict(
            shape=embedding_dim,
        ),
    ),
)
