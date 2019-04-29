# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from typing import Dict

from torch.nn import Module
import numpy as np
from poem.constants import KG_EMBEDDING_MODEL_NAME

from poem.kge_models.kge_models_using_numerical_literals.distmult_literal_e_owa import DistMultLiteral

#: A mapping from KGE model names to KGE model classes
from poem.model_config import ModelConfig
from poem.training_loops.basic_training_loop import TrainingLoop

KGE_MODELS = {
    DistMultLiteral.model_name: DistMultLiteral,
}

def get_kge_model(model_config: ModelConfig) -> Module:
    """Get an instance of a knowledge graph embedding model with the given configuration."""
    kge_model_name = model_config.config[KG_EMBEDDING_MODEL_NAME]
    kge_model_cls = KGE_MODELS.get(kge_model_name)

    if kge_model_cls is None:
        raise ValueError(f'Invalid KGE model name: {kge_model_name}')

    return kge_model_cls(experimental_setup=model_config)

def get_train_loop(model_config: ModelConfig) -> TrainingLoop:
    pass