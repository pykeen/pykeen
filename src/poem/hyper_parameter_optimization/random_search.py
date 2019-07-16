# -*- coding: utf-8 -*-

"""A hyper-parameter optimizer that uses random search."""
import random
from typing import Dict, List, Mapping, Iterable, Any

import numpy as np
import torch
from torch.nn import Module
from tqdm import trange

from poem.evaluation import Evaluator
from poem.hyper_parameter_optimization.HPOptimizer import HPOptimizer, HPOptimizerResult
from poem.instance_creation_factories.instances import Instances
from poem.training_loops import TrainingLoop


class RandomSearch(HPOptimizer):
    """A hyper-parameter optimizer that uses random search."""

    def __init__(
            self,
            mapped_train_triples: np.ndarray,
            mapped_test_triples: np.ndarray,
            entity_to_id: Dict[str:int],
            rel_to_id: Dict[str:int],
            device: torch.device,
    ):
        """."""
        self.mapped_train_triples = mapped_train_triples
        self.mapped_test_triples = mapped_test_triples
        self.entity_to_id = entity_to_id
        self.rel_to_id = rel_to_id
        self.device = device
        # TODO: Set seed?

    def _sample_conv_e_params(self) -> Dict[str, Any]:
        """."""
        pass

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
            model_class,
            training_instances: Instances,
            test_instances: Instances,
            training_loop: TrainingLoop,
            evaluator: Evaluator,
            params_to_values: Mapping[str:List[Any]],
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
            model = type(model_class.__name__, (), **params_to_values)

            # Train model
            trained_model, losses_per_epochs = training_loop.train(
                training_instances=training_instances,
                **current_params_to_values,
            )

            # Evaluate model
            metric_results = evaluator.evaluate(triples=test_instances.instances)
            eval_summaries.append(metric_results)

            trained_kge_models.append(trained_model)
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
