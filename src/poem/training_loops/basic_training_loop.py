# -*- coding: utf-8 -*-

"""Training loops for KGE models using mulitmodal information."""

import logging
from typing import Dict
import torch.nn as nn
import timeit
from abc import ABC, abstractmethod
import numpy as np

from poem.instance_creation_factories.triples_factory import Instances
from poem.model_config import ModelConfig

log = logging.getLogger(__name__)

class TrainingLoop(ABC):
    """."""

    def __init__(self, model_config: ModelConfig, kge_model: nn.Module, instances: Instances):
        self.config = model_config.config
        self.kge_model = kge_model
        self.losses_per_epochs = []
        self.instances = instances
        self.all_entities = np.array(list(self.instances.entity_to_id.values()))

    @abstractmethod
    def train(self):
        pass