# -*- coding: utf-8 -*-

"""Implementation of the basic pipeline."""

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn

from poem.basic_utils import is_evaluation_requested
from poem.constants import EXECUTION_MODE, TRAINING_MODE, HPO_MODE, SEED, CWA, OWA, TEST_SET_PATH, TEST_SET_RATIO, \
    EVALUATOR, RANK_BASED_EVALUATOR
from poem.evaluation.abstract_evaluator import AbstractEvalutor
from poem.evaluation.ranked_based_evaluator import RankBasedEvaluator
from poem.instance_creation_factories.instances import MultimodalInstances
from poem.instance_creation_factories.triples_factory import TriplesFactory, Instances
from poem.instance_creation_factories.utils import get_factory
from poem.kge_models.utils import get_kge_model
from poem.model_config import ModelConfig
from poem.training_loops.basic_training_loop import TrainingLoop
from poem.training_loops.owa_training_loop import OWATrainingLoop

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


@dataclass
class EvaluatorConfig:
    """."""
    config: Dict
    kge_model: nn.Module
    entity_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]
    training_triples: np.ndarray = None


class Pipeline():
    """."""

    KG_ASSUMPTION_TO_TRAINING_LOOP = {
        CWA: None,
        OWA: OWATrainingLoop
    }

    EVALUATORS = {
        RANK_BASED_EVALUATOR: RankBasedEvaluator,
    }

    def __init__(self, config: Dict, training_instances: Optional[Instances] = None,
                 test_instances: Optional[Instances] = None):
        self.config = config
        self.model_config: ModelConfig = None
        self.instance_factory: TriplesFactory = None
        self.training_instances = training_instances
        self.test_instances = test_instances
        self.has_preprocessed_instances = self.training_instances is True

        # Set random generators
        torch.manual_seed(config[SEED])
        np.random.seed(config[SEED])

    @property
    def is_evaluation_requested(self):
        return TEST_SET_PATH in self.config or TEST_SET_RATIO in self.config

    def _create_model_config(self) -> ModelConfig:
        """"""

        multimodal_data = None

        if isinstance(self.training_instances, MultimodalInstances):
            multimodal_data = self.training_instances.multimodal_data

        model_config = ModelConfig(config=self.config, multimodal_data=multimodal_data)
        return model_config

    def _preprocess_train_triples(self) -> Instances:
        """"""

        self.instance_factory = get_factory(self.config)
        return self.instance_factory.create_instances()

    def _proprcess_train_and_test_triples(self) -> (Instances, Instances):
        """"""
        self.instance_factory = get_factory(self.config)
        return self.instance_factory.create_train_and_test_instances()

    def preprocess(self) -> Union[Tuple[Instances, Instances], Instances]:
        """Create instances."""

        if self.has_preprocessed_instances:
            raise Warning("Instances will be created, although already provided")

        if self.is_evaluation_requested:
            return self._proprcess_train_and_test_triples()
        else:
            return self._preprocess_train_triples()

    def _get_evaluator(self, kge_model) -> AbstractEvalutor:
        """."""
        evaluator: AbstractEvalutor = self.config.get(EVALUATOR)
        eval_config = EvaluatorConfig(config=self.config,
                                      entity_to_id=self.training_instances.entity_to_id,
                                      relation_to_id=self.training_instances.relation_to_id,
                                      kge_model=kge_model,
                                      training_triples=self.training_instances.instances)

        return evaluator(evaluator_config=eval_config)

    def _get_training_loop(self, model_config: ModelConfig, kge_model: nn.Module, instances: Instances) -> TrainingLoop:
        """Get training loop."""
        training_loop = self.KG_ASSUMPTION_TO_TRAINING_LOOP[kge_model.kg_assumption]

        return training_loop(model_config=model_config, kge_model=kge_model, instances=instances)

    def _perform_only_training(self):
        """"""
        if self.has_preprocessed_instances is False:
            self.training_instances = self.preprocess()

    def run(self):
        """."""

        if EXECUTION_MODE not in self.config:
            raise KeyError()

        exec_mode = self.config[EXECUTION_MODE]

        if exec_mode != TRAINING_MODE and exec_mode != HPO_MODE:
            raise ValueError()

        if self.has_preprocessed_instances is False:
            self.training_instances = self.preprocess()
        if exec_mode == TRAINING_MODE:
            kge_model, losses_per_epochs = self.train()

            if is_evaluation_requested(config=self.config):
                # eval
                metric_results = self.evaluate(kge_model=kge_model, test_triples=self.test_instances.instances)

        elif exec_mode == HPO_MODE:
            self.perform_hpo()

    def train(self):
        """."""
        self.model_config = self._create_model_config()
        kge_model = get_kge_model(model_config=self.model_config)
        train_loop = self._get_training_loop(model_config=self.model_config,
                                             kge_model=kge_model,
                                             instances=self.training_instances)
        # Train the model based on the defined training loop
        kge_model, losses_per_epochs = train_loop.train()

        return kge_model, losses_per_epochs

    def perform_hpo(self):
        """."""

    def evaluate(self, kge_model, test_triples):
        """."""
        evaluator = self._get_evaluator(kge_model=kge_model)
        metric_results = evaluator.evaluate(test_triples=test_triples)

        return metric_results

# if __name__ == '__main__':
#     p = '/Users/mali/PycharmProjects/LiteralE/data/FB15k/literals/numerical_literals.tsv.txt'
#     t = '/Users/mali/PycharmProjects/LiteralE/data/FB15k/test.txt'
#     config = {
#         KG_EMBEDDING_MODEL_NAME: DISTMULT_LITERAL_NAME_OWA,
#         TRAINING_SET_PATH: t,
#         PATH_TO_NUMERIC_LITERALS: p,
#         KG_ASSUMPTION: OWA,
#         SEED: 2
#
#     }
#     # preprocess
#     pipeline = Pipeline(config=config)
#     instances = pipeline.preprocess()
#     print(instances)
