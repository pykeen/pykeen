"""Tests for training with PyTorch Lightning."""

import itertools

import pytest

from pykeen import models
from pykeen.datasets import EagerDataset, get_dataset
from pykeen.typing import TRAINING
from tests.utils import needs_packages

try:
    from pykeen.contrib.lightning import lit_module_resolver, lit_pipeline

    LIT_MODULES = lit_module_resolver.lookup_dict.keys()
except ImportError:
    LIT_MODULES = []
    lit_pipeline = None


EMBEDDING_DIM = 8
# TODO: this could be shared with the model tests
MODEL_CONFIGURATIONS = {
    models.AutoSF: {"embedding_dim": EMBEDDING_DIM},
    models.BoxE: {"embedding_dim": EMBEDDING_DIM},
    # fixme: CompGCN leads to an autograd runtime error...
    # models.CompGCN: dict(embedding_dim=EMBEDDING_DIM),
    models.ComplEx: {"embedding_dim": EMBEDDING_DIM},
    models.ConvE: {"embedding_dim": EMBEDDING_DIM},
    models.ConvKB: {"embedding_dim": EMBEDDING_DIM, "num_filters": 2},
    models.CP: {"embedding_dim": EMBEDDING_DIM, "rank": 3},
    models.CrossE: {"embedding_dim": EMBEDDING_DIM},
    models.DistMA: {"embedding_dim": EMBEDDING_DIM},
    models.DistMult: {"embedding_dim": EMBEDDING_DIM},
    models.ERMLP: {"embedding_dim": EMBEDDING_DIM},
    models.ERMLPE: {"embedding_dim": EMBEDDING_DIM},
    # FixedModel: dict(embedding_dim=EMBEDDING_DIM),
    models.HolE: {"embedding_dim": EMBEDDING_DIM},
    models.InductiveNodePiece: {"embedding_dim": EMBEDDING_DIM},
    models.InductiveNodePieceGNN: {"embedding_dim": EMBEDDING_DIM},
    models.KG2E: {"embedding_dim": EMBEDDING_DIM},
    models.MuRE: {"embedding_dim": EMBEDDING_DIM},
    models.NodePiece: {"embedding_dim": EMBEDDING_DIM},
    models.NTN: {"embedding_dim": EMBEDDING_DIM},
    models.PairRE: {"embedding_dim": EMBEDDING_DIM},
    models.ProjE: {"embedding_dim": EMBEDDING_DIM},
    models.QuatE: {"embedding_dim": EMBEDDING_DIM},
    models.RESCAL: {"embedding_dim": EMBEDDING_DIM},
    models.RGCN: {"embedding_dim": EMBEDDING_DIM},
    models.RotatE: {"embedding_dim": EMBEDDING_DIM},
    models.SE: {"embedding_dim": EMBEDDING_DIM},
    models.SimplE: {"embedding_dim": EMBEDDING_DIM},
    models.TorusE: {"embedding_dim": EMBEDDING_DIM},
    models.TransD: {"embedding_dim": EMBEDDING_DIM},
    models.TransE: {"embedding_dim": EMBEDDING_DIM},
    models.TransF: {"embedding_dim": EMBEDDING_DIM},
    models.TransH: {"embedding_dim": EMBEDDING_DIM},
    models.TransR: {"embedding_dim": EMBEDDING_DIM, "relation_dim": 3},
    models.TuckER: {"embedding_dim": EMBEDDING_DIM},
    models.UM: {"embedding_dim": EMBEDDING_DIM},
}
TEST_CONFIGURATIONS = (
    (model, model_config, lit)
    for (model, model_config), lit in itertools.product(MODEL_CONFIGURATIONS.items(), LIT_MODULES)
)


# test combinations of models with training loops
@needs_packages("pytorch_lightning")
@pytest.mark.parametrize(("model", "model_kwargs", "training_loop"), TEST_CONFIGURATIONS)
def test_lit_training(model, model_kwargs, training_loop):
    """Test training models with PyTorch Lightning."""
    # some models require inverse relations
    create_inverse_triples = model is not models.RGCN
    dataset = get_dataset(dataset="nations", dataset_kwargs={"create_inverse_triples": create_inverse_triples})

    # some model require access to the training triples
    if "triples_factory" in model_kwargs:
        model_kwargs["triples_factory"] = dataset.training

    # inductive models require an inductive mode to be set, and an inference factory to be passed
    if issubclass(model, models.InductiveNodePiece):
        # fake an inference factory
        model_kwargs["inference_factory"] = dataset.training
        mode = TRAINING
    else:
        mode = None

    lit_pipeline(
        training_loop=training_loop,
        training_loop_kwargs={
            "model": model,
            "dataset": dataset,
            "model_kwargs": model_kwargs,
            "batch_size": 8,
            "mode": mode,
        },
        trainer_kwargs={
            # automatically choose accelerator
            "accelerator": "auto",
            # defaults to TensorBoard; explicitly disabled here
            "logger": False,
            # disable checkpointing
            "enable_checkpointing": False,
            # fast run
            "max_epochs": 2,
        },
    )


@needs_packages("pytorch_lightning")
def test_lit_pipeline_with_dataset_without_validation():
    """Test training on a dataset without validation triples."""
    dataset = get_dataset(dataset="nations")
    dataset = EagerDataset(training=dataset.training, testing=dataset.testing, metadata=dataset.metadata)
    lit_pipeline(
        training_loop="slcwa",
        training_loop_kwargs={
            "model": "transe",
            "dataset": dataset,
        },
        trainer_kwargs={
            # automatically choose accelerator
            "accelerator": "auto",
            # defaults to TensorBoard; explicitly disabled here
            "logger": False,
            # disable checkpointing
            "enable_checkpointing": False,
            # fast run
            "max_epochs": 2,
        },
    )
