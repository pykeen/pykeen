# -*- coding: utf-8 -*-

"""Script for initializing the knowledge graph embedding models."""

from typing import Dict, Mapping

import torch.optim as optim

from pykeen.constants import (
    ADAGRAD_OPTIMIZER_NAME, ADAM_OPTIMIZER_NAME, LEARNING_RATE, OPTMIZER_NAME, SGD_OPTIMIZER_NAME,
)

__all__ = [
    'OPTIMIZERS',
    'get_optimizer',
]

OPTIMIZERS: Mapping = {
    SGD_OPTIMIZER_NAME: optim.SGD,
    ADAGRAD_OPTIMIZER_NAME: optim.Adagrad,
    ADAM_OPTIMIZER_NAME: optim.Adam,
}


def get_optimizer(config: Dict, kge_model):
    """Get an optimizer for the given knowledge graph embedding model."""
    optimizer_name = config[OPTMIZER_NAME]
    optimizer_cls = OPTIMIZERS.get(optimizer_name)

    if optimizer_cls is None:
        raise ValueError(f'invalid optimizer name: {optimizer_name}')

    parameters = filter(lambda p: p.requires_grad, kge_model.parameters())

    return optimizer_cls(parameters, lr=config[LEARNING_RATE])
