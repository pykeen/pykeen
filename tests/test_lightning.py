"""Tests for training with PyTorch Lightning."""

import itertools

import pytest

from pykeen.contrib.lightning import lit_module_resolver, lit_pipeline
from pykeen.datasets import get_dataset
from pykeen.models import *

EMBEDDING_DIM = 8
MODEL_CONFIGURATIONS = {
    AutoSF: dict(embedding_dim=EMBEDDING_DIM),
    BoxE: dict(embedding_dim=EMBEDDING_DIM),
    CompGCN: dict(embedding_dim=EMBEDDING_DIM),
    ComplEx: dict(embedding_dim=EMBEDDING_DIM),
    ConvE: dict(embedding_dim=EMBEDDING_DIM),
    ConvKB: dict(embedding_dim=EMBEDDING_DIM, num_filters=2),
    CP: dict(embedding_dim=EMBEDDING_DIM, rank=3),
    CrossE: dict(embedding_dim=EMBEDDING_DIM),
    DistMA: dict(embedding_dim=EMBEDDING_DIM),
    DistMult: dict(embedding_dim=EMBEDDING_DIM),
    ERMLP: dict(embedding_dim=EMBEDDING_DIM),
    ERMLPE: dict(embedding_dim=EMBEDDING_DIM),
    # FixedModel: dict(embedding_dim=EMBEDDING_DIM),
    HolE: dict(embedding_dim=EMBEDDING_DIM),
    InductiveNodePiece: dict(embedding_dim=EMBEDDING_DIM),
    InductiveNodePieceGNN: dict(embedding_dim=EMBEDDING_DIM),
    KG2E: dict(embedding_dim=EMBEDDING_DIM),
    MuRE: dict(embedding_dim=EMBEDDING_DIM),
    NodePiece: dict(embedding_dim=EMBEDDING_DIM),
    NTN: dict(embedding_dim=EMBEDDING_DIM),
    PairRE: dict(embedding_dim=EMBEDDING_DIM),
    ProjE: dict(embedding_dim=EMBEDDING_DIM),
    QuatE: dict(embedding_dim=EMBEDDING_DIM),
    RESCAL: dict(embedding_dim=EMBEDDING_DIM),
    RGCN: dict(embedding_dim=EMBEDDING_DIM),
    RotatE: dict(embedding_dim=EMBEDDING_DIM),
    SE: dict(embedding_dim=EMBEDDING_DIM),
    SimplE: dict(embedding_dim=EMBEDDING_DIM),
    TorusE: dict(embedding_dim=EMBEDDING_DIM),
    TransD: dict(embedding_dim=EMBEDDING_DIM),
    TransE: dict(embedding_dim=EMBEDDING_DIM),
    TransF: dict(embedding_dim=EMBEDDING_DIM),
    TransH: dict(embedding_dim=EMBEDDING_DIM),
    TransR: dict(embedding_dim=EMBEDDING_DIM, relation_dim=3),
    TuckER: dict(embedding_dim=EMBEDDING_DIM),
    UM: dict(embedding_dim=EMBEDDING_DIM),
}
TEST_CONFIGURATIONS = (
    (model, model_config, lit)
    for (model, model_config), lit in itertools.product(
        MODEL_CONFIGURATIONS.items(), lit_module_resolver.lookup_dict.keys()
    )
)


# test combinations of models with training loops
@pytest.mark.parametrize(("model", "model_kwargs", "training_loop"), TEST_CONFIGURATIONS)
def test_lit_training(model, model_kwargs, training_loop):
    """Test training models with PyTorch Lightning."""
    # some models require inverse relations
    create_inverse_triples = model is not RGCN
    dataset = get_dataset(dataset="nations", dataset_kwargs=dict(create_inverse_triples=create_inverse_triples))
    # some model require access to the training triples
    if "triples_factory" in model_kwargs:
        model_kwargs["triples_factory"] = dataset.training
    if issubclass(model, InductiveNodePiece):
        # fake an inference factory
        model_kwargs["inference_factory"] = dataset.training
    lit_pipeline(
        training_loop=training_loop,
        training_loop_kwargs=dict(
            model=model,
            # use a small configuration for testing
            # TODO: this does not properly work for all models
            dataset=dataset,
            model_kwargs=model_kwargs,
            batch_size=8,
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
