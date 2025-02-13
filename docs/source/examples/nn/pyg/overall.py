"""Example for using PyTorch Geometric.

Combine static label-based entity features with a trainable GCN encoder for entity representations, with learned
embeddings for relation representations and a DistMult interaction function.
"""

from pykeen.datasets import get_dataset
from pykeen.models import ERModel
from pykeen.nn.init import LabelBasedInitializer
from pykeen.pipeline import pipeline
from pykeen.triples.triples_factory import TriplesFactory

dataset = get_dataset(
    dataset="nations",
    dataset_kwargs=dict(create_inverse_triples=True),
)
triples_factory = dataset.training
# build initializer with encoding of entity labels
assert isinstance(triples_factory, TriplesFactory)
entity_initializer = LabelBasedInitializer.from_triples_factory(
    triples_factory=triples_factory,
    for_entities=True,
)
(embedding_dim,) = entity_initializer.tensor.shape[1:]
pipeline(
    dataset=dataset,
    model=ERModel,
    model_kwargs=dict(
        interaction="distmult",
        entity_representations="SimpleMessagePassing",
        entity_representations_kwargs=dict(
            triples_factory=triples_factory,
            base_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                trainable=False,
            ),
            layers=["GCN"] * 2,
            layers_kwargs=dict(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
            ),
        ),
        relation_representations_kwargs=dict(
            shape=embedding_dim,
        ),
    ),
)
