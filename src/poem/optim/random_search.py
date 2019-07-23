# -*- coding: utf-8 -*-

"""A hyper-parameter optimizer that uses random search."""

import random
from typing import Any, Dict, Iterable, List, Mapping, Type

import numpy as np
import torch
from torch.nn import Module
from tqdm import trange

from poem.evaluation import Evaluator
from poem.instance_creation_factories.instances import Instances
from poem.models import BaseModule
from poem.training import TrainingLoop
from poem.utils import get_params_requiring_grad
from .hyper_parameter_optimizer import HPOptimizer, HPOptimizerResult

__all__ = [
    'RandomSearch',
]


class RandomSearch(HPOptimizer):
    """A hyper-parameter optimizer that uses random search."""

    def __init__(
            self,
            model_class: Type[BaseModule],
            optimizer_class: Type[torch.optim.Optimizer],
            entity_to_id: Dict[str, int],
            rel_to_id: Dict[str, int],
            training_loop: TrainingLoop,
            evaluator: Evaluator,
    ):
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.entity_to_id = entity_to_id
        self.rel_to_id = rel_to_id
        self.training_loop = training_loop
        self.evaluator = evaluator

        # TODO: Set seed?

    def _sample_conv_e_params(self) -> Dict[str, Any]:
        pass

    def extract_constructor_arguments(self, params_to_values):
        """Extract params required to initialize model."""
        constructor_args = {}
        for p in self.model_class.get_model_params():
            if p in params_to_values:
                constructor_args[p] = params_to_values[p]

        return constructor_args

    def _sample_parameter_value(self, parameter_to_values: Mapping[int, Iterable[Any]]) -> Mapping[int, Any]:
        """Randomly subsample a dictionary whose values are iterable."""
        return {
            parameter: (
                random.choice(values)
                if isinstance(values, list) else
                values
            )
            for parameter, values in parameter_to_values.items()
        }

    def optimize_hyperparams(
            self,
            training_instances: Instances,
            test_triples: np.ndarray,
            params_to_values: Mapping[str, List[Any]],
            k_evaluation: int = 10,
            max_iters=2,
    ) -> HPOptimizerResult:
        """Apply random search."""

        trained_kge_models: List[Module] = []
        epoch_losses: List[List[float]] = []
        hits_at_k_evaluations: List[float] = []
        models_params: List[Dict] = []
        eval_summaries: List = []

        # TODO: Add ConvE sampler
        sample_fct = self._sample_parameter_value

        for _ in trange(max_iters, desc='HPO Iteration'):
            current_params_to_values: Dict[str, Any] = sample_fct(params_to_values)
            models_params.append(current_params_to_values)

            constructor_args = self.extract_constructor_arguments(params_to_values=current_params_to_values)
            model = self.model_class(**constructor_args)

            params = get_params_requiring_grad(model)

            # Configure optimizer
            optimizer_arguments = {
                'params': params,
                'lr': params_to_values['learning_rate'],
            }

            optimizer = self.optimizer_class(**optimizer_arguments)

            # Train model
            self.training_loop.model = model
            self.training_loop.optimizer = optimizer
            losses_per_epochs = self.training_loop.train(
                training_instances=training_instances,
                **current_params_to_values,
            )

            # Evaluate model
            self.evaluator.model = model
            metric_results = self.evaluator.evaluate(triples=test_triples)
            eval_summaries.append(metric_results)

            trained_kge_models.append(model)
            epoch_losses.append(losses_per_epochs)

            hits_at_k_evaluation = metric_results.hits_at_k[k_evaluation]
            hits_at_k_evaluations.append(hits_at_k_evaluation)

        index_of_max = int(np.argmax(a=hits_at_k_evaluations))

        return (
            trained_kge_models[index_of_max],
            epoch_losses[index_of_max],
            eval_summaries[index_of_max],
            models_params[index_of_max],
        )
