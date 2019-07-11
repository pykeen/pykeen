# -*- coding: utf-8 -*-

"""A hyper-parameter optimizer that uses random search."""
from typing import Dict, List, Mapping, Iterable, Any

import numpy as np
from torch.nn import Module
import random
from poem.hyper_parameter_optimization.HPOptimizer import HPOptimizer, HPOptimizerResult
import torch
from tqdm import trange


class RandomSearch(HPOptimizer):
    """A hyper-parameter optimizer that uses random search."""

    def __init__(
            self,
            mapped_train_triples: np.ndarray,
            mapped_test_triples: np.ndarray,
            entity_to_id: Dict[str:int],
            rel_to_id: Dict[str:int],
            device: torch.device
    ):
        """."""
        self.mapped_train_triples = mapped_train_triples
        self.mapped_test_triples = mapped_test_triples
        self.entity_to_id = entity_to_id
        self.rel_to_id = rel_to_id
        self.device = device
        #TODO: Set seed?

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

    def optimize_hyperparams(self, params_to_values: Mapping[str:List[Any]], max_iters=2) -> HPOptimizerResult:
        """Apply random search."""

        trained_kge_models: List[Module] = []
        epoch_losses: List[List[float]] = []
        hits_at_k_evaluations: List[float] = []
        entity_to_ids: List[Dict[int, str]] = []
        rel_to_ids: List[Dict[int, str]] = []
        models_params: List[Dict] = []
        eval_summaries: List = []

        for _ in trange(max_iters, desc='HPO Iteration'):
            pass