# -*- coding: utf-8 -*-

import random

import numpy as np

from pykeen.constants import *
from pykeen.hyper_parameter_optimizer.abstract_hyper_params_optimizer import AbstractHPOptimizer
from pykeen.utilities.evaluation_utils.metrics_computations import compute_metrics
from pykeen.utilities.initialization_utils.module_initialization_utils import get_kg_embedding_model
from pykeen.utilities.train_utils import train_model

__all__ = ['RandomSearchHPO']


class RandomSearchHPO(AbstractHPOptimizer):

    def _sample_conv_e_params(self, hyparams_dict):
        kg_model_config = OrderedDict()
        # Sample params which are dependent on each other
        embedding_dimensions = hyparams_dict[EMBEDDING_DIM]
        sampled_index = random.choice(range(len(embedding_dimensions)))
        kg_model_config[EMBEDDING_DIM] = embedding_dimensions[sampled_index]
        kg_model_config[CONV_E_HEIGHT] = hyparams_dict[CONV_E_HEIGHT][sampled_index]
        kg_model_config[CONV_E_WIDTH] = hyparams_dict[CONV_E_WIDTH][sampled_index]
        kg_model_config[CONV_E_KERNEL_HEIGHT] = hyparams_dict[CONV_E_KERNEL_HEIGHT][sampled_index]
        kg_model_config[CONV_E_KERNEL_WIDTH] = hyparams_dict[CONV_E_KERNEL_WIDTH][sampled_index]

        del hyparams_dict[EMBEDDING_DIM]
        del hyparams_dict[CONV_E_HEIGHT]
        del hyparams_dict[CONV_E_WIDTH]
        del hyparams_dict[CONV_E_KERNEL_HEIGHT]
        del hyparams_dict[CONV_E_KERNEL_WIDTH]

        kg_model_config.update(self._sample_params(hyparams_dict))

        return kg_model_config

    def _sample_params(self, hyperparams_dict):
        kg_model_config = OrderedDict()
        for param, values in hyperparams_dict.items():
            if isinstance(values, list):
                kg_model_config[param] = random.choice(values)
            else:
                kg_model_config[param] = values

        return kg_model_config

    def optimize_hyperparams(self, mapped_train_tripels, mapped_test_tripels, entity_to_id, rel_to_id, config,
                             device, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)

        trained_models = []
        eval_results = []
        entity_to_ids = []
        rel_to_ids = []
        models_params = []
        eval_summaries = []
        epoch_losses = []

        config = config.copy()
        max_iters = config[NUM_OF_HPO_ITERS]

        sample_fct = self._sample_conv_e_params if config[
                                                       KG_EMBEDDING_MODEL_NAME] == CONV_E_NAME else self._sample_params

        for _ in range(max_iters):
            eval_summary = OrderedDict()

            # Sample hyper-params
            kg_embedding_model_config = sample_fct(config)
            kg_embedding_model_config[NUM_ENTITIES] = len(entity_to_id)
            kg_embedding_model_config[NUM_RELATIONS] = len(rel_to_id)
            kg_embedding_model_config[SEED] = seed

            # Configure defined model
            kg_embedding_model = get_kg_embedding_model(config=kg_embedding_model_config)

            models_params.append(kg_embedding_model_config)
            entity_to_ids.append(entity_to_id)
            rel_to_ids.append(rel_to_id)

            all_entities = np.array(list(entity_to_id.values()))

            trained_model, epoch_loss = train_model(
                kg_embedding_model=kg_embedding_model,
                all_entities=all_entities,
                learning_rate=kg_embedding_model_config[LEARNING_RATE],
                num_epochs=kg_embedding_model_config[NUM_EPOCHS],
                batch_size=kg_embedding_model_config[BATCH_SIZE],
                pos_triples=mapped_train_tripels,
                device=device,
                seed=seed
            )

            # Evaluate trained model
            mean_rank, hits_at_k = compute_metrics(
                all_entities=all_entities,
                kg_embedding_model=trained_model,
                mapped_train_triples=mapped_train_tripels,
                mapped_test_triples=mapped_test_tripels,
                device=device
            )

            # TODO: Define HPO metric
            eval_summary[MEAN_RANK] = mean_rank
            eval_summary[HITS_AT_K] = hits_at_k
            eval_results.append(hits_at_k[10])
            eval_summaries.append(eval_summary)
            trained_models.append(trained_model)
            epoch_losses.append(epoch_loss)

        index_of_max = np.argmax(a=eval_results)

        return trained_models[index_of_max], \
               epoch_losses[index_of_max], \
               entity_to_ids[index_of_max], \
               rel_to_ids[index_of_max], \
               eval_summaries[index_of_max], \
               models_params[index_of_max]

    @staticmethod
    def run(mapped_train_tripels, mapped_test_tripels, entity_to_id, rel_to_id, config, device, seed):
        hpo = RandomSearchHPO()
        return hpo.optimize_hyperparams(
            mapped_train_tripels=mapped_train_tripels,
            mapped_test_tripels=mapped_test_tripels,
            entity_to_id=entity_to_id,
            rel_to_id=rel_to_id,
            config=config,
            device=device,
            seed=seed
        )
