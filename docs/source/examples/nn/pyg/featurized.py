"""Message passing using relation features."""

from pykeen.datasets import get_dataset
from pykeen.models.nbase import ERModel
from pykeen.nn.pyg import FeaturizedMessagePassingRepresentation
from pykeen.nn.representation import Embedding
from pykeen.pipeline import pipeline

embedding_dim = 64
dataset = get_dataset(dataset="nations")
# create embedding matrix for relation representations
relations = Embedding(max_id=dataset.num_relations, embedding_dim=embedding_dim)
entities = FeaturizedMessagePassingRepresentation(
    triples_factory=dataset.training,
    base_kwargs=dict(shape=embedding_dim),
    relation_representation=relations,  # re-use relation representation here
    layers="gat",
    layers_kwargs=dict(
        in_channels=embedding_dim,
        out_channels=embedding_dim,
        edge_dim=embedding_dim,  # should match relation dim
    ),
)
result = pipeline(
    dataset=dataset,
    # compose a model with distmult interaction function
    model=ERModel(
        triples_factory=dataset.training,
        entity_representations=entities,
        relation_representations=relations,
        interaction="DistMult",
    ),
)
