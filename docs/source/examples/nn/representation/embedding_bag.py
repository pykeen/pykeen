"""Embedding bag, with 2048-dimensional Morgan boolean fingerprints for molecules."""

import torch

from pykeen.models import ERModel
from pykeen.nn import Embedding, EmbeddingBagRepresentation
from pykeen.triples.generation import generate_triples_factory
from pykeen.typing import FloatTensor

n_entities = 15
n_relations = 3
n_triples = 100
embedding_dim = 32

# mock some triples
triples_factory = generate_triples_factory(n_entities, n_relations, n_triples)

# mock some boolean feature tensor
features = torch.rand(n_entities, embedding_dim) < 0.5

entity_representation = EmbeddingBagRepresentation.from_iter(
    # might need to flatten here
    list(fingerprint.nonzero())
    for fingerprint in features
)

# we're going to use DistMult as the interaction, so
# we need a relation representation of the same size
relation_representation = Embedding(max_id=n_relations, embedding_dim=embedding_dim)

model = ERModel[FloatTensor, FloatTensor, FloatTensor](
    triples_factory=triples_factory,
    interaction="DistMult",
    entity_representations=entity_representation,
    relation_representations=relation_representation,
)
