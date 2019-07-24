# -*- coding: utf-8 -*-

"""Implementation of the basic pipeline."""

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from dataclasses_json import dataclass_json

from poem.constants import (
    CWA, DISTMULT_LITERAL_NAME_OWA, EVALUATOR, EXECUTION_MODE, HPO_MODE, MODEL_NAME,
    NUM_ENTITIES, NUM_RELATIONS, OWA, RANK_BASED_EVALUATOR, SEED, TEST_SET_PATH, TEST_SET_RATIO, TRAINING_MODE,
)
from poem.evaluation import Evaluator, EvaluatorConfig, RankBasedEvaluator
from poem.instance_creation_factories.instances import Instances, MultimodalInstances
from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from poem.models import DistMultLiteral, ModelConfig
from poem.training import OWATrainingLoop, TrainingLoop

log = logging.getLogger(__name__)


@dataclass_json
@dataclass
class EvalResults:
    """Results from computing metrics."""

    mean_rank: float
    hits_at_k: Dict[int, float]


@dataclass
class ExperimentalArtifacts:
    """Contains the experimental artifacts."""
    trained_model: nn.Module
    losses: list
    entities_to_embeddings: Mapping[str, np.ndarray]
    relations_to_embeddings: Mapping[str, np.ndarray]
    entities_to_ids: Mapping[str, int]
    relations_to_ids: Mapping[str, int]


class Helper:
    MODELS = {
        model_cls.model_name: model_cls
        for model_cls in [
            DistMultLiteral,
        ]
    }

    KG_ASSUMPTION_TO_TRAINING_LOOP = {
        CWA: None,
        OWA: OWATrainingLoop,
    }

    EVALUATORS: Mapping[str, Type[Evaluator]] = {
        RANK_BASED_EVALUATOR: RankBasedEvaluator,
    }

    MODEL_NAME_TO_FACTORY = {
        DISTMULT_LITERAL_NAME_OWA: TriplesNumericLiteralsFactory,
    }

    # ---------------------Get pipeline components---------------------#
    @staticmethod
    def get_evaluator(
            model: nn.Module,
            config: Dict,
            entity_to_id: Dict,
            relation_to_id: Dict,
            training_triples: np.ndarray,
    ) -> Evaluator:
        evaluator_name = config.get(EVALUATOR)
        if evaluator_name is None:
            raise ValueError(f'Configuration is missing key: {EVALUATOR}')

        evaluator_cls: Type[Evaluator] = Helper.EVALUATORS.get(evaluator_name)
        if evaluator_cls is None:
            raise ValueError(f'Invalid evaluator name: {evaluator_name}')

        evaluator_config = EvaluatorConfig(
            config=config,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            model=model,
            training_triples=training_triples,
        )

        evaluator: Evaluator = evaluator_cls(evaluator_config=evaluator_config)
        return evaluator

    @staticmethod
    def get_training_loop(
            config: Dict,
            model: nn.Module,
            all_entities: np.ndarray,
    ) -> TrainingLoop:
        """Get training loop."""
        training_loop = Helper.KG_ASSUMPTION_TO_TRAINING_LOOP[model.kg_assumption]
        return training_loop(config=config, model=model, all_entities=all_entities)

    @staticmethod
    def get_model(model_config: ModelConfig) -> nn.Module:
        """Get an instance of a knowledge graph embedding model with the given configuration."""
        model_name = model_config.config[MODEL_NAME]
        model_cls = Helper.MODELS.get(model_name)

        if model_cls is None:
            raise ValueError(f'Invalid model name: {model_name}')

        return model_cls(model_config=model_config)

    @staticmethod
    def create_model_config(config, instances: Instances) -> ModelConfig:
        multimodal_data = None

        if isinstance(instances, MultimodalInstances):
            multimodal_data = instances.multimodal_data

        model_config = ModelConfig(config=config, multimodal_data=multimodal_data)
        return model_config

    @staticmethod
    def get_factory(model_name, entity_to_id, relation_to_id) -> TriplesFactory:
        factory = Helper.MODEL_NAME_TO_FACTORY.get(model_name)

        if factory is None:
            raise ValueError(f'invalid factory name: {model_name}')

        return factory(entity_to_id, relation_to_id)

    @staticmethod
    def preprocess_train_triples(model_name, entity_to_id, relation_to_id, kg_assumption) -> Instances:
        # FIXME
        instance_factory = Helper.get_factory(config)
        return instance_factory.create_instances()

    @staticmethod
    def preprocess_train_and_test_triples(config) -> (Instances, Instances):
        instance_factory = Helper.get_factory(config)
        return instance_factory.create_train_and_test_instances()


class Pipeline:

    def __init__(
            self,
            config: Dict,
            training_instances: Optional[Instances] = None,
            test_instances: Optional[Instances] = None,
    ) -> None:
        self.config = config
        self.model_config: ModelConfig = None
        self.instance_factory: TriplesFactory = None
        self.training_loop = None
        self.evaluator = None
        self.training_instances = training_instances
        self.test_instances = test_instances
        self.has_preprocessed_instances = self.training_instances is True

        # Set random generators
        torch.manual_seed(config[SEED])
        np.random.seed(config[SEED])

    @property
    def _is_evaluation_requested(self):
        return TEST_SET_PATH in self.config or TEST_SET_RATIO in self.config

    # --------Pipeline's instance methods--------#
    def preprocess(self) -> Union[Tuple[Instances, Instances], Instances]:
        """Create instances."""

        if self.has_preprocessed_instances:
            raise Warning("Instances will be created, although already provided")

        if self._is_evaluation_requested:
            return Helper.preprocess_train_and_test_triples(config=self.config)
        else:
            return Helper.preprocess_train_triples(config=self.config)

    def _perform_only_training(self):
        if not self.has_preprocessed_instances:
            self.training_instances = self.preprocess()

    def _train(self, model, training_instances):
        self.training_loop = Helper.get_training_loop(
            config=model.model_config.config,
            model=model,
            all_entities=training_instances,
        )
        # Train the model based on the defined training loop
        _, losses_per_epochs = self.training_loop.train(
        )

        return model, losses_per_epochs

    def _perform_hpo(self):
        pass

    def _evaluate(self, model, test_triples, entity_to_id, relation_to_id, training_triples):
        self.evaluator = Helper.get_evaluator(
            model=model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            training_triples=training_triples,
        )
        metric_results = self.evaluator.evaluate(triples=test_triples)

        return metric_results

    def run(self):
        if EXECUTION_MODE not in self.config:
            raise KeyError()

        exec_mode = self.config[EXECUTION_MODE]

        if exec_mode != TRAINING_MODE and exec_mode != HPO_MODE:
            raise ValueError()

        if self.has_preprocessed_instances is False:
            self.training_instances = self.preprocess()

        # set number of entities and relations
        self.config[NUM_ENTITIES] = len(self.training_instances.entity_to_id)
        # There are models such as UM that don't consider relations
        if self.training_instances.relation_to_id is not None:
            self.config[NUM_RELATIONS] = len(self.training_instances.relation_to_id)

        if exec_mode == TRAINING_MODE:
            self.model_config = self.create_model_config(config=self.config, instances=self.training_instances)
            model = self.get_model(model_config=self.model_config)
            _, losses_per_epochs = self._train(
                model=model,
                training_instances=self.training_instances,
            )
            if self._is_evaluation_requested:
                # eval
                metric_results = self._evaluate(model=model, test_triples=self.test_instances.instances)

        elif exec_mode == HPO_MODE:
            self._perform_hpo()
