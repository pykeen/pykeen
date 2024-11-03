"""Estimating token diversity for NodePiece."""

from pykeen.datasets import get_dataset
from pykeen.models import NodePiece

model = NodePiece(
    triples_factory=get_dataset(dataset="CodexSmall", dataset_kwargs=dict(create_inverse_triples=True)).training,
    tokenizers=["AnchorTokenizer", "RelationTokenizer"],
    num_tokens=[20, 12],
    embedding_dim=8,
    interaction="distmult",
    entity_initializer="xavier_uniform_",
)
print(model.entity_representations[0].estimate_diversity())
