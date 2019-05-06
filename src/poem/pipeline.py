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
    EVALUATOR, RANK_BASED_EVALUATOR, KG_EMBEDDING_MODEL_NAME, DISTMULT_LITERAL_NAME_OWA, TRAINING_SET_PATH, \
    PATH_TO_NUMERIC_LITERALS, KG_ASSUMPTION, MARGIN_LOSS
from poem.evaluation.abstract_evaluator import AbstractEvalutor, EvaluatorConfig
from poem.evaluation.ranked_based_evaluator import RankBasedEvaluator
from poem.instance_creation_factories.instances import MultimodalInstances
from poem.instance_creation_factories.triples_factory import TriplesFactory, Instances
from poem.instance_creation_factories.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from poem.kge_models.kge_models_using_numerical_literals.distmult_literal_e_owa import DistMultLiteral
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


class Pipeline():
    """."""

    KGE_MODELS = {
        DistMultLiteral.model_name: DistMultLiteral,
    }

    KG_ASSUMPTION_TO_TRAINING_LOOP = {
        CWA: None,
        OWA: OWATrainingLoop
    }

    EVALUATORS = {
        RANK_BASED_EVALUATOR: RankBasedEvaluator,
    }

    KGE_MODEL_NAME_TO_FACTORY = {
        DISTMULT_LITERAL_NAME_OWA: TriplesNumericLiteralsFactory
    }

    def __init__(self, config: Dict, training_instances: Optional[Instances] = None,
                 test_instances: Optional[Instances] = None):
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

    # ---------------------Get pipeline components---------------------#
    @staticmethod
    def get_evaluator(kge_model, config, entity_to_id, relation_to_id, training_triples) -> AbstractEvalutor:
        """."""
        evaluator: AbstractEvalutor = config.get(EVALUATOR)
        eval_config = EvaluatorConfig(config=config,
                                      entity_to_id=entity_to_id,
                                      relation_to_id=relation_to_id,
                                      kge_model=kge_model,
                                      training_triples=training_triples)

        return evaluator(evaluator_config=eval_config)

    @staticmethod
    def get_training_loop(model_config: ModelConfig, kge_model: nn.Module, instances: Instances) -> TrainingLoop:
        """Get training loop."""
        training_loop = Pipeline.KG_ASSUMPTION_TO_TRAINING_LOOP[kge_model.kg_assumption]

        return training_loop(model_config=model_config, kge_model=kge_model, instances=instances)

    @staticmethod
    def get_kge_model(model_config: ModelConfig) -> nn.Module:
        """Get an instance of a knowledge graph embedding model with the given configuration."""
        kge_model_name = model_config.config[KG_EMBEDDING_MODEL_NAME]
        kge_model_cls = Pipeline.KGE_MODELS.get(kge_model_name)

        if kge_model_cls is None:
            raise ValueError(f'Invalid KGE model name: {kge_model_name}')

        return kge_model_cls(model_config=model_config)

    @staticmethod
    def create_model_config(config, instances: Instances) -> ModelConfig:
        """"""

        multimodal_data = None

        if isinstance(instances, MultimodalInstances):
            multimodal_data = instances.multimodal_data

        model_config = ModelConfig(config=config, multimodal_data=multimodal_data)
        return model_config

    @staticmethod
    def get_factory(config) -> TriplesFactory:
        """."""
        kge_model_name = config[KG_EMBEDDING_MODEL_NAME]
        factory = Pipeline.KGE_MODEL_NAME_TO_FACTORY.get(kge_model_name)

        if factory is None:
            raise ValueError(f'invalid factory name: {kge_model_name}')

        return factory(config)

    @staticmethod
    def preprocess_train_triples(config) -> Instances:
        """"""

        instance_factory = Pipeline.get_factory(config)
        return instance_factory.create_instances()

    @staticmethod
    def proprcess_train_and_test_triples(config) -> (Instances, Instances):
        """"""
        instance_factory = Pipeline.get_factory(config)
        return instance_factory.create_train_and_test_instances()

    # -------- Pipeline's instance methods--------#
    def preprocess(self) -> Union[Tuple[Instances, Instances], Instances]:
        """Create instances."""

        if self.has_preprocessed_instances:
            raise Warning("Instances will be created, although already provided")

        if self._is_evaluation_requested:
            return self.proprcess_train_and_test_triples(config=self.config)
        else:
            return self.preprocess_train_triples(config=self.config)

    def _perform_only_training(self):
        """"""
        if self.has_preprocessed_instances is False:
            self.training_instances = self.preprocess()

    def train(self, kge_model, training_instances):
        """."""
        self.training_loop = self.get_training_loop(model_config=kge_model.model_config,
                                                    kge_model=kge_model,
                                                    instances=training_instances)
        # Train the model based on the defined training loop
        fitted_kge_model, losses_per_epochs = self.training_loop.train()

        return fitted_kge_model, losses_per_epochs

    def perform_hpo(self):
        """."""

    def evaluate(self, kge_model, test_triples):
        """."""
        self.evaluator = self.get_evaluator(kge_model=kge_model)
        metric_results = self.evaluator.evaluate(test_triples=test_triples)

        return metric_results

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
            self.model_config = self.create_model_config(config=self.config, instances=self.training_instances)
            kge_model = self.get_kge_model(model_config=self.model_config)
            fitted_kge_model, losses_per_epochs = self.train(kge_model=kge_model,
                                                             training_instances=self.training_instances)

            if is_evaluation_requested(config=self.config):
                # eval
                metric_results = self.evaluate(kge_model=fitted_kge_model, test_triples=self.test_instances.instances)

        elif exec_mode == HPO_MODE:
            self.perform_hpo()


if __name__ == '__main__':
    p = '/Users/mali/PycharmProjects/LiteralE/data/FB15k/literals/numerical_literals.txt'
    t = '/Users/mali/PycharmProjects/LiteralE/data/FB15k/test.txt'
    config = {
        KG_EMBEDDING_MODEL_NAME: DISTMULT_LITERAL_NAME_OWA,
        TRAINING_SET_PATH: t,
        PATH_TO_NUMERIC_LITERALS: p,
        KG_ASSUMPTION: OWA,
        MARGIN_LOSS: 1,

        SEED: 2

    }
    # preprocess
    pipeline = Pipeline(config=config)
    instances = pipeline.preprocess()
    model_config = pipeline.create_model_config(config=config, instances=instances)
    kge_model = pipeline.get_kge_model(model_config=model_config)
    fitted_kge_model, losses = pipeline.train()
    print(fitted_kge_model)
    print(losses)
