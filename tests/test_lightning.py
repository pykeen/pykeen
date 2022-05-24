"""Tests for training with PyTorch Lightning."""

import itertools

import pytest

try:
    from pykeen.contrib.lightning import lit_module_resolver

    LIT_MODULES = lit_module_resolver.lookup_dict.keys()
except ImportError:
    LIT_MODULES = []
from pykeen import models
from pykeen.datasets import get_dataset
from pykeen.typing import TRAINING

EMBEDDING_DIM = 8
# TODO: this could be shared with the model tests
MODEL_CONFIGURATIONS = {
    models.AutoSF: dict(embedding_dim=EMBEDDING_DIM),
    models.BoxE: dict(embedding_dim=EMBEDDING_DIM),
    models.CompGCN: dict(embedding_dim=EMBEDDING_DIM),
    models.ComplEx: dict(embedding_dim=EMBEDDING_DIM),
    models.ConvE: dict(embedding_dim=EMBEDDING_DIM),
    models.ConvKB: dict(embedding_dim=EMBEDDING_DIM, num_filters=2),
    models.CP: dict(embedding_dim=EMBEDDING_DIM, rank=3),
    models.CrossE: dict(embedding_dim=EMBEDDING_DIM),
    models.DistMA: dict(embedding_dim=EMBEDDING_DIM),
    models.DistMult: dict(embedding_dim=EMBEDDING_DIM),
    models.ERMLP: dict(embedding_dim=EMBEDDING_DIM),
    models.ERMLPE: dict(embedding_dim=EMBEDDING_DIM),
    # FixedModel: dict(embedding_dim=EMBEDDING_DIM),
    models.HolE: dict(embedding_dim=EMBEDDING_DIM),
    models.InductiveNodePiece: dict(embedding_dim=EMBEDDING_DIM),
    models.InductiveNodePieceGNN: dict(embedding_dim=EMBEDDING_DIM),
    models.KG2E: dict(embedding_dim=EMBEDDING_DIM),
    models.MuRE: dict(embedding_dim=EMBEDDING_DIM),
    models.NodePiece: dict(embedding_dim=EMBEDDING_DIM),
    models.NTN: dict(embedding_dim=EMBEDDING_DIM),
    models.PairRE: dict(embedding_dim=EMBEDDING_DIM),
    models.ProjE: dict(embedding_dim=EMBEDDING_DIM),
    models.QuatE: dict(embedding_dim=EMBEDDING_DIM),
    models.RESCAL: dict(embedding_dim=EMBEDDING_DIM),
    models.RGCN: dict(embedding_dim=EMBEDDING_DIM),
    models.RotatE: dict(embedding_dim=EMBEDDING_DIM),
    models.SE: dict(embedding_dim=EMBEDDING_DIM),
    models.SimplE: dict(embedding_dim=EMBEDDING_DIM),
    models.TorusE: dict(embedding_dim=EMBEDDING_DIM),
    models.TransD: dict(embedding_dim=EMBEDDING_DIM),
    models.TransE: dict(embedding_dim=EMBEDDING_DIM),
    models.TransF: dict(embedding_dim=EMBEDDING_DIM),
    models.TransH: dict(embedding_dim=EMBEDDING_DIM),
    models.TransR: dict(embedding_dim=EMBEDDING_DIM, relation_dim=3),
    models.TuckER: dict(embedding_dim=EMBEDDING_DIM),
    models.UM: dict(embedding_dim=EMBEDDING_DIM),
}
TEST_CONFIGURATIONS = (
    (model, model_config, lit)
    for (model, model_config), lit in itertools.product(MODEL_CONFIGURATIONS.items(), LIT_MODULES)
)


# test combinations of models with training loops
@pytest.mark.parametrize(("model", "model_kwargs", "training_loop"), TEST_CONFIGURATIONS)
def test_lit_training(model, model_kwargs, training_loop):
    """Test training models with PyTorch Lightning."""
    from pykeen.contrib.lightning import lit_pipeline

    # some models require inverse relations
    create_inverse_triples = model is not models.RGCN
    dataset = get_dataset(dataset="nations", dataset_kwargs=dict(create_inverse_triples=create_inverse_triples))

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
        training_loop_kwargs=dict(
            model=model,
            dataset=dataset,
            model_kwargs=model_kwargs,
            batch_size=8,
            mode=mode,
        ),
        trainer_kwargs=dict(
            # automatically choose accelerator
            accelerator="auto",
            # defaults to TensorBoard; explicitly disabled here
            logger=False,
            # disable checkpointing
            enable_checkpointing=False,
            # fast run
            max_epochs=2,
        ),
    )
