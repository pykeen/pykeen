# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from torch.nn import Module

from poem.constants import KG_EMBEDDING_MODEL_NAME, CWA, OWA, KG_ASSUMPTION, SGD_OPTIMIZER_NAME, ADAGRAD_OPTIMIZER_NAME, \
    ADAM_OPTIMIZER_NAME, OPTMIZER_NAME, LEARNING_RATE
from poem.instance_creation_factories.triples_factory import Instances
from poem.kge_models.kge_models_using_numerical_literals.distmult_literal_e_owa import DistMultLiteral
#: A mapping from KGE model names to KGE model classes
from poem.model_config import ModelConfig
from poem.training_loops.basic_training_loop import TrainingLoop
from poem.training_loops.owa_training_loop import OWATrainingLoop
from typing import  Dict
import torch.optim as optim

KGE_MODELS = {
    DistMultLiteral.model_name: DistMultLiteral,
}

TRAIN_LOOPS = {
    CWA: None,
    OWA: OWATrainingLoop
}

OPTIMIZERS: Dict = {
    SGD_OPTIMIZER_NAME: optim.SGD,
    ADAGRAD_OPTIMIZER_NAME: optim.Adagrad,
    ADAM_OPTIMIZER_NAME: optim.Adam,
}


def get_kge_model(model_config: ModelConfig) -> Module:
    """Get an instance of a knowledge graph embedding model with the given configuration."""
    kge_model_name = model_config.config[KG_EMBEDDING_MODEL_NAME]
    kge_model_cls = KGE_MODELS.get(kge_model_name)

    if kge_model_cls is None:
        raise ValueError(f'Invalid KGE model name: {kge_model_name}')

    return kge_model_cls(experimental_setup=model_config)


def get_training_loop(model_config: ModelConfig, kge_model: Module, instances: Instances) -> TrainingLoop:
    """Get training training loop for defined experiment."""
    traiing_loop = TRAIN_LOOPS.get(KG_ASSUMPTION)
    return traiing_loop(model_config, kge_model, instances)

def get_optimizer(config: Dict, kge_model):
    """Get an optimizer for the given knowledge graph embedding model."""
    optimizer_name = config.get(OPTMIZER_NAME)
    optimizer_cls = OPTIMIZERS.get(optimizer_name)

    if optimizer_cls is None:
        raise ValueError(f'invalid optimizer name: {optimizer_name}')

    parameters = filter(lambda p: p.requires_grad, kge_model.parameters())

    return optimizer_cls(parameters, lr=config.get(LEARNING_RATE))
