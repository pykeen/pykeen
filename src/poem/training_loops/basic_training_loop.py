# -*- coding: utf-8 -*-

"""Training loops for KGE models using mulitmodal information."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Mapping

import numpy as np
import torch.nn as nn
from torch import optim

from poem.constants import SGD_OPTIMIZER_NAME, ADAGRAD_OPTIMIZER_NAME, ADAM_OPTIMIZER_NAME, OPTMIZER_NAME, LEARNING_RATE
from poem.instance_creation_factories.triples_factory import Instances
from poem.model_config import ModelConfig

log = logging.getLogger(__name__)


class TrainingLoop(ABC):
    """."""

    OPTIMIZERS: Mapping = {
        SGD_OPTIMIZER_NAME: optim.SGD,
        ADAGRAD_OPTIMIZER_NAME: optim.Adagrad,
        ADAM_OPTIMIZER_NAME: optim.Adam,
    }

    def __init__(self, config: Dict, kge_model: nn.Module, all_entities: np.ndarray = None):
        self.config = config
        self.kge_model = kge_model
        self.losses_per_epochs = []
        self.all_entities = all_entities

    @abstractmethod
    def train(self, training_instances: Instances):
        pass

    def get_optimizer(self, config: Dict, kge_model):
        """Get an optimizer for the given knowledge graph embedding model."""
        optimizer_name = config[OPTMIZER_NAME]
        optimizer_cls = self.OPTIMIZERS.get(optimizer_name)

        if optimizer_cls is None:
            raise ValueError(f'invalid optimizer name: {optimizer_name}')

        parameters = filter(lambda p: p.requires_grad, kge_model.parameters())

        return optimizer_cls(parameters, lr=config[LEARNING_RATE])
