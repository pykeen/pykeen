# -*- coding: utf-8 -*-
import logging
from collections import OrderedDict

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from evaluation_methods.mean_rank_evaluator import MeanRankEvaluator
from hyper_parameter_optimizer.random_search_optimizer import RandomSearchHPO
from utilities.constants import KG_EMBEDDING_MODEL, NUM_ENTITIES, NUM_RELATIONS, PREFERRED_DEVICE, \
    GPU, HPO, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE
from utilities.initialization_utils.module_initialization_utils import get_kg_embedding_model
from utilities.train_utils import train
from utilities.triples_creation_utils.instance_creation_utils import create_mapped_triples

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Pipeline(object):

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and self.config[PREFERRED_DEVICE] == GPU else 'cpu')

    def start_hpo(self):
        assert HPO in self.config
        return self._start_pipeline(is_hpo_mode=True)

    def start_training(self):
        return self._start_pipeline(is_hpo_mode=False)

    def _start_pipeline(self, is_hpo_mode):
        """
        :return:
        """

        # Initialize reader
        log.info("-------------Read Corpus-------------")

        # TODO: Adapt
        evaluator = MeanRankEvaluator()  # get_evaluator(config=evaluator_config)
        path_to_train_data = self.config['training_set_path']

        if is_hpo_mode:
            hp_optimizer = RandomSearchHPO(evaluator=evaluator)
            trained_model, train_entity_to_id, train_rel_to_id, eval_result, metric_string, params = hp_optimizer.optimize_hyperparams(
                path_to_train_data, self.config, self.device, self.seed)
        else:
            pos_triples = np.loadtxt(fname=path_to_train_data, dtype=str, comments='@Comment@ Subject Predicate Object')
            # neg_triples = create_negative_triples(seed=self.seed, pos_triples=pos_triples,
            #                                       filter_neg_triples=False)

            if 'validation_set_path' not in self.config:
                ratio_test_data = self.config['validation_set_ratio']
                train_pos, test_pos = train_test_split(pos_triples, test_size=ratio_test_data, random_state=self.seed)
            else:
                train_pos = pos_triples
                test_pos = np.loadtxt(fname=self.config['validation_set_path'], dtype=str,
                                      comments='@Comment@ Subject Predicate Object')

            mapped_pos_tripels, train_entity_to_id, train_rel_to_id = create_mapped_triples(train_pos)
            # mapped_neg_triples, _, _ = create_mapped_triples(pos_triples, entity_to_id=train_entity_to_id,
            #                                                  rel_to_id=train_rel_to_id)

            # Initialize KG embedding model

            kb_embedding_model_config = self.config[KG_EMBEDDING_MODEL]
            kb_embedding_model_config[NUM_ENTITIES] = len(train_entity_to_id)
            kb_embedding_model_config[NUM_RELATIONS] = len(train_rel_to_id)
            kg_embedding_model = get_kg_embedding_model(config=kb_embedding_model_config)

            batch_size = kb_embedding_model_config[BATCH_SIZE]
            num_epochs = kb_embedding_model_config[NUM_EPOCHS]
            lr = kb_embedding_model_config[LEARNING_RATE]
            params = kb_embedding_model_config

            log.info("-------------Train KG Embeddings-------------")
            print(batch_size)
            trained_model = train(kg_embedding_model=kg_embedding_model, learning_rate=lr, num_epochs=num_epochs,
                                  batch_size=batch_size, pos_triples=mapped_pos_tripels,
                                  device=self.device, seed=self.seed)

            log.info("-------------Start Evaluation-------------")
            # Initialize KG evaluator
            mapped_pos_test_tripels, _, _ = create_mapped_triples(test_pos)
            eval_result, metric_string = evaluator.start_evaluation(test_data=mapped_pos_test_tripels,
                                                                    kg_embedding_model=trained_model)

        # Prepare Output
        eval_summary = OrderedDict()
        eval_summary[metric_string] = eval_result
        id_to_entity = {value: key for key, value in train_entity_to_id.items()}
        id_to_rel = {value: key for key, value in train_rel_to_id.items()}

        entity_to_embedding = {id_to_entity[id]: embedding.detach().cpu().numpy() for id, embedding in
                               enumerate(trained_model.entities_embeddings.weight)}
        relation_to_embedding = {id_to_rel[id]: embedding.detach().cpu().numpy() for id, embedding in
                                 enumerate(trained_model.relation_embeddings.weight)}

        return trained_model, eval_summary, entity_to_embedding, relation_to_embedding, params
