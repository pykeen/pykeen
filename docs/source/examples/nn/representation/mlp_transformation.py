"""Demonstrate applying a learnable MLP transformation on top of a representation."""

import torch

from pykeen.models import ERModel
from pykeen.nn import Embedding, MLPTransformedRepresentation
from pykeen.triples.generation import generate_triples_factory
from pykeen.typing import FloatTensor

n_entities = 15
n_relations = 3
n_triples = 100
features_dim = 256
target_dim = 32

# mock some triples
triples_factory = generate_triples_factory(n_entities, n_relations, n_triples)

# mock some feature tensor
features = torch.rand(n_entities, features_dim)

base_representation = Embedding.from_pretrained(features)

# this embedding is learned on top of the base representation
entity_representation = MLPTransformedRepresentation(base=base_representation, output_dim=target_dim)

# we're going to use DistMult as the interaction, so
# we need a relation representation of the same size
relation_representation = Embedding(max_id=n_relations, embedding_dim=features_dim)

model = ERModel[FloatTensor, FloatTensor, FloatTensor](
    triples_factory=triples_factory,
    interaction="DistMult",
    entity_representations=entity_representation,
    relation_representations=relation_representation,
)
