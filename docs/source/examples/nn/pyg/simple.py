"""An example for using simple message passing, ignoring edge types.

We create a two-layer GCN on top of an Embedding.
"""

from pykeen.datasets import get_dataset
from pykeen.models import ERModel
from pykeen.nn.pyg import SimpleMessagePassingRepresentation
from pykeen.pipeline import pipeline

embedding_dim = 64
dataset = get_dataset(dataset="nations")
entities = SimpleMessagePassingRepresentation(
    triples_factory=dataset.training,
    base_kwargs=dict(shape=embedding_dim),
    layers=["gcn"] * 2,
    layers_kwargs=dict(in_channels=embedding_dim, out_channels=embedding_dim),
)
result = pipeline(
    dataset=dataset,
    # compose a model with distmult interaction function
    model=ERModel(
        triples_factory=dataset.training,
        entity_representations=entities,
        relation_representations_kwargs=dict(embedding_dim=embedding_dim),  # use embedding with same dimension
        interaction="DistMult",
    ),
)
