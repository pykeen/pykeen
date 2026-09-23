"""Setup an inductive node piece model."""

from pykeen.datasets.inductive.ilp_teru import InductiveFB15k237
from pykeen.losses import NSSALoss
from pykeen.models.inductive import InductiveNodePiece

dataset = InductiveFB15k237(version="v1", create_inverse_triples=True)

model = InductiveNodePiece(
    triples_factory=dataset.transductive_training,  # training factory, used to tokenize training nodes
    inference_factory=dataset.inductive_inference,  # inference factory, used to tokenize inference nodes
    num_tokens=12,  # length of a node hash - how many unique relations per node will be used
    aggregation="mlp",  # aggregation function, defaults to an MLP, can be any PyTorch function
    loss=NSSALoss(margin=15),  # dummy loss
    random_seed=42,
)
