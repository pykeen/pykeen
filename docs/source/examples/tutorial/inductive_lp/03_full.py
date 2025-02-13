"""Full example for inductive link prediction."""

from torch.optim import Adam

from pykeen.datasets.inductive.ilp_teru import InductiveFB15k237
from pykeen.evaluation.rank_based_evaluator import SampledRankBasedEvaluator
from pykeen.losses import NSSALoss
from pykeen.models.inductive import InductiveNodePieceGNN
from pykeen.stoppers import EarlyStopper
from pykeen.training import SLCWATrainingLoop

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
# ensure that the model is on the right device
model = model.to(model.device)

optimizer = Adam(params=model.parameters(), lr=0.0005)

training_loop = SLCWATrainingLoop(
    triples_factory=dataset.transductive_training,  # training triples
    model=model,
    optimizer=optimizer,
    negative_sampler_kwargs=dict(num_negs_per_pos=32),
    mode="training",  # necessary to specify for the inductive mode - training has its own set of nodes
)

# Validation and Test evaluators use a restricted protocol ranking against 50 random negatives
assert dataset.inductive_validation is not None
valid_evaluator = SampledRankBasedEvaluator(
    mode="validation",  # necessary to specify for the inductive mode - this will use inference nodes
    evaluation_factory=dataset.inductive_validation,  # validation triples to predict
    additional_filter_triples=dataset.inductive_inference.mapped_triples,  # filter out true inference triples
)

# According to the original code
# https://github.com/kkteru/grail/blob/2a3dffa719518e7e6250e355a2fb37cd932de91e/test_ranking.py#L526-L529
# test filtering uses only the inductive_inference split and does not include inductive_validation triples
# If you use the full RankBasedEvaluator, both inductive_inference and inductive_validation triples
# must be added to the additional_filter_triples
test_evaluator = SampledRankBasedEvaluator(
    mode="testing",  # necessary to specify for the inductive mode - this will use inference nodes
    evaluation_factory=dataset.inductive_testing,  # test triples to predict
    additional_filter_triples=dataset.inductive_inference.mapped_triples,  # filter out true inference triples
)

early_stopper = EarlyStopper(
    model=model,
    training_triples_factory=dataset.inductive_inference,
    evaluation_triples_factory=dataset.inductive_validation,
    frequency=1,
    patience=100000,  # for test reasons, turn it off
    result_tracker=None,
    evaluation_batch_size=256,
    evaluator=valid_evaluator,
)

# Training starts here
training_loop.train(
    triples_factory=dataset.transductive_training,
    stopper=early_stopper,
    num_epochs=100,
)

# Test evaluation
result = test_evaluator.evaluate(
    model=model,
    mapped_triples=dataset.inductive_testing.mapped_triples,
    additional_filter_triples=dataset.inductive_inference.mapped_triples,
    batch_size=256,
)

# print final results
print(result.to_flat_dict())
