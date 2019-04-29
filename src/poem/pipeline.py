# -*- coding: utf-8 -*-

"""Implementation of the basic pipeline."""

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import torch.nn as nn

from poem import model_config
from poem.basic_utils import is_evaluation_requested
from poem.constants import EXECUTION_MODE, TRAINING_MODE, HPO_MODE, KG_EMBEDDING_MODEL_NAME, DISTMULT_LITERAL_NAME_OWA, \
    PATH_TO_NUMERIC_LITERALS, SEED
from poem.instance_creation_factories.triples_factory import TriplesFactory, Instances
from poem.instance_creation_factories.utils import get_factory
from poem.kge_models.utils import get_kge_model, get_training_loop
from poem.model_config import ModelConfig
import torch

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

    def __init__(self, config: Dict, instances: Optional[Instances]):
        self.config = config
        self.model_config: ModelConfig = None
        self.instance_factory: TriplesFactory = None
        self.instances = instances
        self.has_preprocessed_instances = self.instances is True

        # Set random generators
        torch.manual_seed(config[SEED])
        np.random.seed(config[SEED])

    def run(self):
        """."""

        if EXECUTION_MODE not in self.config:
            raise KeyError()

        exec_mode = self.config[EXECUTION_MODE]

        if exec_mode != TRAINING_MODE and exec_mode != HPO_MODE:
            raise ValueError()

        if self.has_preprocessed_instances is False:
            self.instances = self.preprocess()
        if exec_mode == TRAINING_MODE:
            kge_model, losses_per_epochs = self.train()

            if is_evaluation_requested(config=self.config):
                # eval
                pass

        elif exec_mode == HPO_MODE:
            self.perform_hpo()

    def preprocess(self):
        """Create instances."""

        if self.has_preprocessed_instances:
            raise Warning("Instances will be created, although already provided")

        self.instance_factory = get_factory(self.config)
        return self.instance_factory.create_instances()

    def train(self):
        """."""
        self.model_config = ModelConfig(config=self.config,
                                        multimodal_data=self.instances.multimodal_data,
                                        has_multimodal_data=self.instances.has_multimodal_data)
        kge_model = get_kge_model(model_config=model_config)
        train_loop = get_training_loop(model_config=self.model_config, kge_model=kge_model, instances=self.instances)
        # Train the model based on the defined training loop
        kge_model, losses_per_epochs = train_loop.train()

        return kge_model, losses_per_epochs

    def perform_hpo(self):
        """."""

    def evaluate(self):
        """."""


if __name__ == '__main__':
    config = {
        KG_EMBEDDING_MODEL_NAME: DISTMULT_LITERAL_NAME_OWA,
        PATH_TO_NUMERIC_LITERALS: ''

    }

    # preprocess
    pipeline = Pipeline(config=config)
    instances = pipeline.preprocess()
