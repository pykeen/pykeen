# -*- coding: utf-8 -*-

"""Implementation of the basic pipeline."""

import logging
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import torch.nn as nn

from poem import model_config
from poem.constants import EXECUTION_MODE, TRAINING_MODE, HPO_MODE
from poem.instance_creation_factories.triples_factory import TriplesFactory, Instances
from poem.instance_creation_factories.utils import get_factory
from poem.kge_models.utils import get_kge_model, get_training_loop
from poem.model_config import ModelConfig
from poem.training_loops.basic_training_loop import TrainingLoop

log = logging.getLogger(__name__)


@dataclass
class EvalResults:
    """Results from computing metrics."""

    mean_rank: float
    hits_at_k: Dict[int, float]


@dataclass
class ExperimentalArtifacts():
    """Contains the experimental artifacts."""
    trained_kge_model: nn.Module
    losses: list
    entities_to_embeddings: Mapping[str, np.ndarray]
    relations_to_embeddings: Mapping[str, np.ndarray]
    entities_to_ids: Mapping[str, int]
    relations_to_ids: Mapping[str, int]


@dataclass
class ExperimentalArtifactsContainingEvalResults(ExperimentalArtifacts):
    """."""
    eval_results: EvalResults


class Pipeline():
    """."""

    def __init__(self, config):
        self.config: Dict = config
        self.model_config: ModelConfig = None
        self.instance_factory: TriplesFactory = None
        self.instances: Instances = None

    def run(self):
        """."""

        # Step 1: Create instances
        self.instances = self.preprocess()

        if EXECUTION_MODE not in self.config:
            raise KeyError()

        exec_mode = self.config[EXECUTION_MODE]

        if exec_mode != TRAINING_MODE and exec_mode != HPO_MODE:
            raise ValueError()

        # Step 2: Train in training or HPO mode
        if exec_mode == TRAINING_MODE:
            self.train()
        elif exec_mode == HPO_MODE:
            self.perform_hpo()

    def preprocess(self):
        """Create instances."""
        self.instance_factory = get_factory(self.config)
        return self.instance_factory.create_instances()

    def train(self):
        """."""
        self.model_config = ModelConfig(config=self.config,
                                   multimodal_data=self.instances.multimodal_data,
                                   has_multimodal_data= self.instances.has_multimodal_data )
        kge_model = get_kge_model(model_config=model_config)
        train_loop = get_training_loop(model_config=self.model_config, kge_model=kge_model, instances=self.instances)
        # Train the model based on the defined training loop
        kge_model, losses_per_epochs = train_loop.train()

        return kge_model, losses_per_epochs

    def perform_hpo(self):
        """."""

    def evaluate(self):
        """."""


def run(config: Dict) -> ExperimentalArtifacts:
    """."""

    # Determine execution mode: Training (Evaluation), HPO
    # Determine training approach: OWA or CWA
    # Determines how to create training instances

    '''
    Step 1: Load data
    Step 2: Create instances based on assumption and model
    '''
    NotImplemented
