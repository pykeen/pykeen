"""Demonstrate creating a model with a feature-enriched embedding."""

import torch

from pykeen.models import ERModel
from pykeen.nn import Embedding, FeatureEnrichedEmbedding
from pykeen.triples.generation import generate_triples_factory
from pykeen.typing import FloatTensor

n_entities = 15
n_relations = 3
n_triples = 100
embedding_dim = 32

# mock some triples
triples_factory = generate_triples_factory(n_entities, n_relations, n_triples)

# mock some feature tensor
features = torch.rand(n_entities, embedding_dim)

# this embedding is a combination of the features
# and a learnable embedding of the same shape
entity_representation = FeatureEnrichedEmbedding(features)

# we're going to use DistMult as the interaction, so
# we need a relation representation of the same size
relation_representation = Embedding(max_id=n_relations, embedding_dim=embedding_dim)

model = ERModel[FloatTensor, FloatTensor, FloatTensor](
    triples_factory=triples_factory,
    interaction="DistMult",
    entity_representations=entity_representation,
    relation_representations=relation_representation,
)
