# -*- coding: utf-8 -*-

"""Training loops for KGE models using multi-modal information."""

import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Mapping, Tuple, Type

import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.optim import Adagrad, Adam, Optimizer, SGD

from ..constants import ADAGRAD_OPTIMIZER_NAME, ADAM_OPTIMIZER_NAME, SGD_OPTIMIZER_NAME
from ..instance_creation_factories.instances import Instances

__all__ = [
    'OPTIMIZERS',
    'TrainingLoop',
]

log = logging.getLogger(__name__)

OPTIMIZERS: Mapping[str, Type[Optimizer]] = {
    SGD_OPTIMIZER_NAME: SGD,
    ADAGRAD_OPTIMIZER_NAME: Adagrad,
    ADAM_OPTIMIZER_NAME: Adam,
}


def get_params(module: nn.Module) -> Iterable[Tensor]:
    return filter(lambda p: p.requires_grad, module.parameters())


class TrainingLoop(ABC):
    def __init__(
            self,
            kge_model: nn.Module,
            optimizer,
            all_entities: np.ndarray = None,
    ) -> None:
        self.kge_model = kge_model
        self.optimizer = optimizer
        self.losses_per_epochs = []
        self.all_entities = all_entities

    @property
    def device(self):
        return self.kge_model.device

    @abstractmethod
    def train(
            self,
            training_instances: Instances,
            num_epochs,
            batch_size,
            learning_rate,
    ) -> Tuple[nn.Module, List[float]]:
        """Train the KGE model.

        :return: A pair of the KGE model and the losses per epoch.
        """

    @staticmethod
    def get_optimizer(
            optimizer_name: str,
            kge_model: nn.Module,
            lr: float,
    ):
        """Get an optimizer for the given knowledge graph embedding model."""
        optimizer_cls: Type[Optimizer] = OPTIMIZERS.get(optimizer_name)

        if optimizer_cls is None:
            raise ValueError(f'invalid optimizer name: {optimizer_name}')

        params = get_params(kge_model)

        return optimizer_cls(
            params=params,
            lr=lr,
        )
