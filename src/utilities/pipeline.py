# -*- coding: utf-8 -*-
import logging
from collections import OrderedDict

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from hyper_parameter_optimizer.random_search_optimizer import RandomSearchHPO
from utilities.constants import KG_EMBEDDING_MODEL, NUM_ENTITIES, NUM_RELATIONS, PREFERRED_DEVICE, \
    GPU, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, TRAINING_SET_PATH, TEST_SET_PATH, TEST_SET_RATIO, \
    MEAN_RANK, HITS_AT_K, CPU
from utilities.evaluation_utils.compute_metrics import compute_metrics
from utilities.initialization_utils.module_initialization_utils import get_kg_embedding_model
from utilities.train_utils import train_model
from utilities.triples_creation_utils.instance_creation_utils import create_mapped_triples, create_mappings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Pipeline(object):

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and self.config[PREFERRED_DEVICE] == GPU else CPU)

    def start_hpo(self):
        return self._start_pipeline(is_hpo_mode=True)

    def start_training(self):
        return self._start_pipeline(is_hpo_mode=False)

    def _start_pipeline(self, is_hpo_mode):
        """
        :return:
        """

        # TODO: Adapt
        path_to_train_data = self.config[TRAINING_SET_PATH]

        pos_triples = np.loadtxt(fname=path_to_train_data, dtype=str, comments='@Comment@ Subject Predicate Object')
        has_test_set = True

        if TEST_SET_PATH in self.config:
            train_pos = pos_triples
            test_pos = np.loadtxt(fname=self.config[TEST_SET_PATH], dtype=str,
                                  comments='@Comment@ Subject Predicate Object')
        else:
            train_pos = pos_triples
            ratio_test_data = self.config[TEST_SET_RATIO]
            train_pos, test_pos = train_test_split(pos_triples, test_size=ratio_test_data, random_state=self.seed)
            # has_test_set = False

        # Create entity and relation mapping
        all_triples = np.concatenate([train_pos, test_pos], axis=0)
        entity_to_id, rel_to_id = create_mappings(triples=all_triples)
        mapped_pos_train_tripels, _, _ = create_mapped_triples(triples=train_pos, entity_to_id=entity_to_id,
                                                               rel_to_id=rel_to_id)

        all_entities = np.array(list(entity_to_id.values()))

        if is_hpo_mode:
            hp_optimizer = RandomSearchHPO()

            trained_model, loss_per_epoch, entity_to_embedding, relation_to_embedding, eval_summary, metric_string, params = hp_optimizer.optimize_hyperparams(
                train_pos, test_pos,
                entity_to_id,
                rel_to_id,
                mapped_pos_train_tripels,
                self.config,
                self.device,
                self.seed)

        else:
            # Initialize KG embedding model
            kb_embedding_model_config = self.config[KG_EMBEDDING_MODEL]
            kb_embedding_model_config[NUM_ENTITIES] = len(entity_to_id)
            kb_embedding_model_config[NUM_RELATIONS] = len(rel_to_id)
            kb_embedding_model_config[PREFERRED_DEVICE] = self.device
            kg_embedding_model = get_kg_embedding_model(config=kb_embedding_model_config)

            batch_size = kb_embedding_model_config[BATCH_SIZE]
            num_epochs = kb_embedding_model_config[NUM_EPOCHS]
            lr = kb_embedding_model_config[LEARNING_RATE]
            params = kb_embedding_model_config

            log.info("-------------Train KG Embeddings-------------")
            trained_model, loss_per_epoch = train_model(kg_embedding_model=kg_embedding_model,
                                                        all_entities=all_entities, learning_rate=lr,
                                                        num_epochs=num_epochs,
                                                        batch_size=batch_size, pos_triples=mapped_pos_train_tripels,
                                                        device=self.device, seed=self.seed)

            eval_summary = None

            if has_test_set:
                log.info("-------------Start Evaluation-------------")
                # Initialize KG evaluator
                mapped_pos_test_tripels, _, _ = create_mapped_triples(triples=test_pos, entity_to_id=entity_to_id,
                                                                      rel_to_id=rel_to_id)

                eval_summary = OrderedDict()
                mean_rank, hits_at_k = compute_metrics(all_entities=all_entities,
                                                       kg_embedding_model=kg_embedding_model,
                                                       triples=mapped_pos_test_tripels, device=self.device)

                eval_summary[MEAN_RANK] = mean_rank
                eval_summary[HITS_AT_K] = hits_at_k

        # Prepare Output
        id_to_entity = {value: key for key, value in entity_to_id.items()}
        id_to_rel = {value: key for key, value in rel_to_id.items()}
        entity_to_embedding = {id_to_entity[id]: embedding.detach().cpu().numpy() for id, embedding in
                               enumerate(trained_model.entities_embeddings.weight)}
        relation_to_embedding = {id_to_rel[id]: embedding.detach().cpu().numpy() for id, embedding in
                                 enumerate(trained_model.relation_embeddings.weight)}

        return trained_model, loss_per_epoch, eval_summary, entity_to_embedding, relation_to_embedding, params
