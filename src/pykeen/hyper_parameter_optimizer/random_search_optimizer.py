# -*- coding: utf-8 -*-

"""A hyper-parameter optimizer that uses random search."""

import random
from typing import Any, List, Optional, Tuple

from torch.nn import Module
from tqdm import trange

from pykeen.constants import *
from pykeen.hyper_parameter_optimizer.abstract_hyper_params_optimizer import AbstractHPOptimizer
from pykeen.kge_models import get_kge_model
from pykeen.utilities.evaluation_utils.metrics_computations import compute_metric_results
from pykeen.utilities.train_utils import train_kge_model

__all__ = ['RandomSearchHPO']

OptimizeResult = Tuple[Module, List[float], Any, Any, Any, Any]


class RandomSearchHPO(AbstractHPOptimizer):
    """A hyper-parameter optimizer that uses random search."""

    def _sample_conv_e_params(self, hyperparams_dict) -> Dict[str, Any]:
        kg_model_config = OrderedDict()
        # Sample params which are dependent on each other
        embedding_dimensions = hyperparams_dict[EMBEDDING_DIM]
        sampled_index = random.choice(range(len(embedding_dimensions)))
        kg_model_config[EMBEDDING_DIM] = embedding_dimensions[sampled_index]
        kg_model_config[CONV_E_HEIGHT] = hyperparams_dict[CONV_E_HEIGHT][sampled_index]
        kg_model_config[CONV_E_WIDTH] = hyperparams_dict[CONV_E_WIDTH][sampled_index]
        kg_model_config[CONV_E_KERNEL_HEIGHT] = hyperparams_dict[CONV_E_KERNEL_HEIGHT][sampled_index]
        kg_model_config[CONV_E_KERNEL_WIDTH] = hyperparams_dict[CONV_E_KERNEL_WIDTH][sampled_index]

        del hyperparams_dict[EMBEDDING_DIM]
        del hyperparams_dict[CONV_E_HEIGHT]
        del hyperparams_dict[CONV_E_WIDTH]
        del hyperparams_dict[CONV_E_KERNEL_HEIGHT]
        del hyperparams_dict[CONV_E_KERNEL_WIDTH]

        kg_model_config.update(self._sample_parameter_value(hyperparams_dict))

        return kg_model_config

    def optimize_hyperparams(self,
                             mapped_train_triples,
                             mapped_test_triples,
                             entity_to_id,
                             rel_to_id,
                             config,
                             device,
                             seed: Optional[int] = None,
                             k_evaluation: int = 10) -> OptimizeResult:
        """"""
        if seed is not None:
            # FIXME np.random is not used
            np.random.seed(seed=seed)

        trained_kge_models: List[Module] = []
        epoch_losses: List[List[float]] = []
        hits_at_k_evaluations: List[float] = []
        entity_to_ids: List[Dict[int, str]] = []
        rel_to_ids: List[Dict[int, str]] = []
        models_params: List[Dict] = []
        eval_summaries: List = []

        config = config.copy()
        max_iters = config[NUM_OF_HPO_ITERS]

        sample_fct = (
            self._sample_conv_e_params
            if config[KG_EMBEDDING_MODEL_NAME] == CONV_E_NAME else
            self._sample_parameter_value
        )

        for _ in trange(max_iters, desc='HPO Iteration'):
            # Sample hyper-params
            kge_model_config: Dict[str, Any] = sample_fct(config)
            kge_model_config[NUM_ENTITIES]: int = len(entity_to_id)
            kge_model_config[NUM_RELATIONS]: int = len(rel_to_id)
            kge_model_config[SEED]: int = seed

            # Configure defined model
            kge_model: Module = get_kge_model(config=kge_model_config)

            models_params.append(kge_model_config)
            entity_to_ids.append(entity_to_id)
            rel_to_ids.append(rel_to_id)

            all_entities = np.array(list(entity_to_id.values()))

            trained_kge_model, epoch_loss = train_kge_model(
                kge_model=kge_model,
                all_entities=all_entities,
                learning_rate=kge_model_config[LEARNING_RATE],
                num_epochs=kge_model_config[NUM_EPOCHS],
                batch_size=kge_model_config[BATCH_SIZE],
                pos_triples=mapped_train_triples,
                device=device,
                seed=seed,
                tqdm_kwargs=dict(leave=False),
            )

            # Evaluate trained model
            mean_rank, hits_at_k = compute_metric_results(
                all_entities=all_entities,
                kg_embedding_model=trained_kge_model,
                mapped_train_triples=mapped_train_triples,
                mapped_test_triples=mapped_test_triples,
                device=device,
            )

            # TODO: Define HPO metric
            eval_summary = _make_eval_summary(mean_rank, hits_at_k)
            eval_summaries.append(eval_summary)

            trained_kge_models.append(trained_kge_model)
            epoch_losses.append(epoch_loss)

            hits_at_k_evaluation = hits_at_k[k_evaluation]
            hits_at_k_evaluations.append(hits_at_k_evaluation)

        index_of_max = int(np.argmax(a=hits_at_k_evaluations))

        return (
            trained_kge_models[index_of_max],
            epoch_losses[index_of_max],
            entity_to_ids[index_of_max],
            rel_to_ids[index_of_max],
            eval_summaries[index_of_max],
            models_params[index_of_max],
        )

    @classmethod
    def run(cls,
            mapped_train_triples: np.ndarray,
            mapped_test_triples: np.ndarray,
            entity_to_id: Dict[int, str],
            rel_to_id: Dict[int, str],
            config: Dict,
            device,
            seed) -> OptimizeResult:
        return cls().optimize_hyperparams(
            mapped_train_triples=mapped_train_triples,
            mapped_test_triples=mapped_test_triples,
            entity_to_id=entity_to_id,
            rel_to_id=rel_to_id,
            config=config,
            device=device,
            seed=seed,
        )


def _make_eval_summary(mean_rank: float, hits_at_k: Dict[int, float]):
    eval_summary = OrderedDict()
    eval_summary[MEAN_RANK]: float = mean_rank
    eval_summary[HITS_AT_K]: Dict[int, float] = hits_at_k
    return eval_summary
