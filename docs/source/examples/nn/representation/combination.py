"""Demonstrate creating a model with a combine representation."""

import torch

from pykeen.models import ERModel
from pykeen.nn import CombinedRepresentation, ConcatProjectionCombination, Embedding
from pykeen.triples.generation import generate_triples_factory
from pykeen.typing import FloatTensor

n_entities = 15
n_relations = 3
n_triples = 100
feature_dim = 35
pre_combination_embedding_dim = 32
post_combination_embedding_dim = 16

# mock some triples
triples_factory = generate_triples_factory(n_entities, n_relations, n_triples)

# mock some feature tensor
features = torch.rand(n_entities, feature_dim)

# note that the embedding doesn't need the same dimension as the
embedding = Embedding(max_id=n_entities, shape=pre_combination_embedding_dim)

# this embedding is a combination of the features and a learnable embedding.
entity_representation = CombinedRepresentation(
    max_id=n_entities,
    base=[embedding, features],
    combination=ConcatProjectionCombination,
    combination_kwargs=dict(output_dim=post_combination_embedding_dim),
)

# we're going to use DistMult as the interaction, so we need a relation
# representation of the same shape as the result of the combine representation
relation_representation = Embedding(max_id=n_relations, embedding_dim=post_combination_embedding_dim)

model = ERModel[FloatTensor, FloatTensor, FloatTensor](
    triples_factory=triples_factory,
    interaction="DistMult",
    entity_representations=entity_representation,
    relation_representations=relation_representation,
)
