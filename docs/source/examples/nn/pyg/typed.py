"""Example for message passing with type information.

Here, we use a one-layer RGCN using the basis decomposition.
"""

from pykeen.datasets import get_dataset
from pykeen.models import ERModel
from pykeen.nn.pyg import TypedMessagePassingRepresentation
from pykeen.pipeline import pipeline

embedding_dim = 64
dataset = get_dataset(dataset="nations")
entities = TypedMessagePassingRepresentation(
    triples_factory=dataset.training,
    base_kwargs=dict(shape=embedding_dim),
    layers="rgcn",
    layers_kwargs=dict(
        in_channels=embedding_dim,
        out_channels=embedding_dim,
        num_bases=2,
        num_relations=dataset.num_relations,
    ),
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
