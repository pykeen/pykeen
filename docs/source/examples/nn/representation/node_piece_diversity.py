"""Estimating token diversity for NodePiece."""

from pykeen.datasets import CoDExSmall
from pykeen.models import NodePiece

dataset = CoDExSmall(create_inverse_triples=True)
model = NodePiece(
    triples_factory=dataset.training,
    tokenizers=["AnchorTokenizer", "RelationTokenizer"],
    num_tokens=[20, 12],
    embedding_dim=8,
    interaction="distmult",
    entity_initializer="xavier_uniform_",
)
print(model.entity_representations[0].estimate_diversity())
