# -*- coding: utf-8 -*-

"""Training loops for KGE models using multi-modal information."""

import logging
from abc import ABC, abstractmethod
from typing import List, Mapping, Tuple, Type

import numpy as np
import torch.nn as nn
from torch.optim import Adagrad, Adam, Optimizer, SGD

from ..constants import ADAGRAD_OPTIMIZER_NAME, ADAM_OPTIMIZER_NAME, SGD_OPTIMIZER_NAME
from ..instance_creation_factories.instances import Instances
from ..utils import get_params

__all__ = [
    'TrainingLoop',
]

log = logging.getLogger(__name__)

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
    ) -> Tuple[nn.Module, List[float]]:
        """Train the KGE model.

        :return: A pair of the KGE model and the losses per epoch.
        """
