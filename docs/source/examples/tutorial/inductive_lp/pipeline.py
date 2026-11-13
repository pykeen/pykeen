"""Train and evaluate an inductive NodePiece model with the pipeline."""

import click

from pykeen.datasets.inductive import InductiveFB15k237
from pykeen.pipeline import pipeline

dataset = InductiveFB15k237(version="v1")
assert dataset.inductive_validation is not None

result = pipeline(
    training=dataset.transductive_training,
    testing=dataset.inductive_testing,
    model="InductiveNodePiece",
    model_kwargs={
        "inference_factory": dataset.inductive_inference,
        "embedding_dim": 32,
        "num_tokens": 4,
        "use_inverse_triples": True,
    },
    training_loop="sLCWA",
    training_loop_kwargs={"mode": "training"},
    training_kwargs={"num_epochs": 5, "batch_size": 128},
    evaluator="RankBasedEvaluator",
    evaluator_kwargs={"mode": "testing"},
    evaluation_kwargs={
        "additional_filter_triples": [
            dataset.inductive_inference.mapped_triples,
            dataset.inductive_validation.mapped_triples,
        ],
        "batch_size": 128,
    },
    device="cpu",
    random_seed=42,
)

click.echo(result.metric_results.to_flat_dict())
