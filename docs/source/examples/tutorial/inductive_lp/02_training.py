"""Train the inductive model."""

from pykeen.datasets.inductive.ilp_teru import InductiveFB15k237
from pykeen.evaluation.rank_based_evaluator import SampledRankBasedEvaluator
from pykeen.training import SLCWATrainingLoop

dataset = InductiveFB15k237(version="v1", create_inverse_triples=True)

model = ...  # model init here, one of InductiveNodePiece
optimizer = ...  # some optimizer

training_loop = SLCWATrainingLoop(
    triples_factory=dataset.transductive_training,  # training triples
    model=model,
    optimizer=optimizer,
    mode="training",  # necessary to specify for the inductive mode - training has its own set of nodes
)

assert dataset.inductive_validation is not None
valid_evaluator = SampledRankBasedEvaluator(
    mode="validation",  # necessary to specify for the inductive mode - this will use inference nodes
    evaluation_factory=dataset.inductive_validation,  # validation triples to predict
    additional_filter_triples=dataset.inductive_inference.mapped_triples,  # filter out true inference triples
)

test_evaluator = SampledRankBasedEvaluator(
    mode="testing",  # necessary to specify for the inductive mode - this will use inference nodes
    evaluation_factory=dataset.inductive_testing,  # test triples to predict
    additional_filter_triples=dataset.inductive_inference.mapped_triples,  # filter out true inference triples
)
