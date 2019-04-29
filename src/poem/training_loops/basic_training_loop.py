# -*- coding: utf-8 -*-

"""Training loops for KGE models using mulitmodal information."""

import logging
from typing import Dict
import torch.nn as nn
import timeit
from abc import ABC, abstractmethod

from poem.model_config import ModelConfig

log = logging.getLogger(__name__)

class TrainingLoop(ABC):
    """."""

    def __init__(self, model_config: ModelConfig, kge_model: nn.Module):
        self.config = model_config.config
        self.kge_model = kge_model

    @abstractmethod
    def train(self):
        pass