# -*- coding: utf-8 -*-

"""A hyper-parameter optimizer that uses random search."""

import random
from typing import Any, Dict, Iterable, List, Mapping, Type

import numpy as np
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from tqdm import trange

from .hyper_parameter_optimizer import HPOptimizer, HPOptimizerResult
from ..evaluation import Evaluator
from ..instance_creation_factories import Instances
from ..models.base import BaseModule
from ..training import TrainingLoop

__all__ = [
    'RandomSearch',
]


class RandomSearch(HPOptimizer):
    """A hyper-parameter optimizer that uses random search."""

    def __init__(
            self,
            model_cls: Type[BaseModule],
            optimizer_cls: Type[Optimizer],
            entity_to_id: Dict[str, int],
            rel_to_id: Dict[str, int],
            training_loop: TrainingLoop,
            evaluator: Evaluator,
    ) -> None:
        """Initialize the random search hyper-parameter optimizer."""
        self.model_cls = model_cls
        self.optimizer_cls = optimizer_cls
        self.entity_to_id = entity_to_id
        self.rel_to_id = rel_to_id
        self.training_loop = training_loop
        self.evaluator = evaluator

        # TODO: Set seed?

    def _sample_conv_e_params(self) -> Dict[str, Any]:
        pass

    def extract_constructor_arguments(self, params_to_values):
        """Extract params required to initialize model."""
        return {
            p: params_to_values[p]
            for p in self.model_cls.get_model_params()
            if p in params_to_values
        }

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
            model = self.model_cls(**constructor_args)

            params = model.get_grad_params()

            # Configure optimizer
            optimizer_arguments = {
                'params': params,
                'lr': params_to_values['learning_rate'],
            }

            optimizer = self.optimizer_cls(**optimizer_arguments)

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
