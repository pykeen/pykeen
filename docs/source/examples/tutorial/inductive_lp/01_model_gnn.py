"""Setup an inductive node piece model with GNN encoder."""

from pykeen.datasets.inductive.ilp_teru import InductiveFB15k237
from pykeen.losses import NSSALoss
from pykeen.models.inductive import InductiveNodePieceGNN

dataset = InductiveFB15k237(version="v1", create_inverse_triples=True)

model = InductiveNodePieceGNN(
    triples_factory=dataset.transductive_training,  # training factory, will be also used for a GNN
    inference_factory=dataset.inductive_inference,  # inference factory, will be used for a GNN
    num_tokens=12,  # length of a node hash - how many unique relations per node will be used
    aggregation="mlp",  # aggregation function, defaults to an MLP, can be any PyTorch function
    loss=NSSALoss(margin=15),  # dummy loss
    random_seed=42,
    gnn_encoder=None,  # defaults to a 2-layer CompGCN with DistMult composition function
)
