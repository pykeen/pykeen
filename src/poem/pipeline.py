# -*- coding: utf-8 -*-

"""Implementation of the basic pipeline."""

import logging
from typing import Dict, Mapping
from dataclasses import dataclass
import torch.nn as nn
import numpy as np

from poem.constants import EXECUTION_MODE, HPO_MODE, TEST_SET_PATH, TEST_SET_RATIO, OWA, KG_ASSUMPTION

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

@property
def is_hpo_mode(config) -> bool:
    """."""
    return config[EXECUTION_MODE] == HPO_MODE

@property
def is_evaluation_requested(config) -> bool:
    return TEST_SET_PATH in config or TEST_SET_RATIO in config

@property
def is_owa(config):
    """."""
    return config[KG_ASSUMPTION] == config[OWA]


def run(config: Dict) -> ExperimentalArtifacts:
    """."""

    # Determine execution mode: Training (Evaluation), HPO
    # Determine training approach: OWA or CWA
      # Determines how to create training instances

    '''
    Step 1: Load data
    Step 2: Create instances based on assumption and model
    '''



