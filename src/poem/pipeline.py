# -*- coding: utf-8 -*-

"""Implementation of the basic pipeline."""

import logging
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import torch.nn as nn

from poem.constants import EXECUTION_MODE, TRAINING_MODE, HPO_MODE
from poem.utils import get_factory

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
        self.config = config
        self.instance_factory = None
        self.instances = None

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
