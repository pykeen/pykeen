# -*- coding: utf-8 -*-

"""Implementation of the basic pipeline."""

import logging
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pykeen.constants import *
from pykeen.hyper_parameter_optimizer.random_search_optimizer import RandomSearchHPO
from pykeen.utilities.evaluation_utils.metrics_computations import compute_metrics
from pykeen.utilities.initialization_utils.module_initialization_utils import get_kg_embedding_model
from pykeen.utilities.train_utils import train_model
from pykeen.utilities.triples_creation_utils.instance_creation_utils import create_mapped_triples, create_mappings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Pipeline(object):
    def __init__(self, config, seed):
        self.config = config
        self.seed = seed
        self.entity_to_id = None
        self.rel_to_id = None
        self.device_name=(
            'cuda:0'
            if torch.cuda.is_available() and self.config[PREFERRED_DEVICE] == GPU else
            CPU
        )
        self.device = torch.device(self.device_name)

    def start(self, path_to_train_data: Optional[str] = None):
        is_hpo_mode = self.config[EXECUTION_MODE] == HPO_MODE
        return self._start_pipeline(is_hpo_mode=is_hpo_mode, path_to_train_data=path_to_train_data)

    @property
    def is_evaluation_required(self) -> bool:
        return TEST_SET_PATH in self.config or TEST_SET_RATIO in self.config

    # TODO: Remove path_to_train_data
    def _start_pipeline(self, is_hpo_mode: bool, path_to_train_data: Optional[str] = None):

        if is_hpo_mode:
            mapped_pos_train_triples, mapped_pos_test_triples = self._get_train_and_test_triples()

            (trained_model,
             loss_per_epoch,
             entity_to_embedding,
             relation_to_embedding,
             eval_summary,
             params) = RandomSearchHPO.run(
                mapped_train_tripels=mapped_pos_train_triples,
                mapped_test_tripels=mapped_pos_test_triples,
                entity_to_id=self.entity_to_id,
                rel_to_id=self.rel_to_id,
                config=self.config,
                device=self.device,
                seed=self.seed
            )

        else:
            # Training Mode
            if self.is_evaluation_required:
                mapped_pos_train_triples, mapped_pos_test_triples = self._get_train_and_test_triples()
            else:
                mapped_pos_train_triples = self._get_train_triples()

            all_entities = np.array(list(self.entity_to_id.values()))

            # Initialize KG embedding model
            self.config[NUM_ENTITIES] = len(self.entity_to_id)
            self.config[NUM_RELATIONS] = len(self.rel_to_id)
            self.config[PREFERRED_DEVICE] = self.device_name
            kg_embedding_model = get_kg_embedding_model(config=self.config)

            batch_size = self.config[BATCH_SIZE]
            num_epochs = self.config[NUM_EPOCHS]
            learning_rate = self.config[LEARNING_RATE]
            params = self.config

            log.info("-------------Train KG Embeddings-------------")
            trained_model, loss_per_epoch = train_model(kg_embedding_model=kg_embedding_model,
                                                        all_entities=all_entities,
                                                        learning_rate=learning_rate,
                                                        num_epochs=num_epochs,
                                                        batch_size=batch_size,
                                                        pos_triples=mapped_pos_train_triples,
                                                        device=self.device,
                                                        seed=self.seed)

            eval_summary = None

            if self.is_evaluation_required:
                log.info("-------------Start Evaluation-------------")

                eval_summary = OrderedDict()
                mean_rank, hits_at_k = compute_metrics(all_entities=all_entities,
                                                       kg_embedding_model=kg_embedding_model,
                                                       mapped_train_triples=mapped_pos_train_triples,
                                                       mapped_test_triples=mapped_pos_test_triples,
                                                       device=self.device,
                                                       filter_neg_triples=self.config[FILTER_NEG_TRIPLES])

                eval_summary[MEAN_RANK] = mean_rank
                eval_summary[HITS_AT_K] = hits_at_k

        # Prepare Output
        id_to_entity = {value: key for key, value in self.entity_to_id.items()}
        id_to_rel = {value: key for key, value in self.rel_to_id.items()}
        entity_to_embedding = {
            id_to_entity[id]: embedding.detach().cpu().numpy()
            for id, embedding in enumerate(trained_model.entity_embeddings.weight)
        }

        if self.config[KG_EMBEDDING_MODEL_NAME] in [SE_NAME, UM_NAME]:
            relation_to_embedding = None
        else:
            relation_to_embedding = {
                id_to_rel[id]: embedding.detach().cpu().numpy()
                for id, embedding in enumerate(trained_model.relation_embeddings.weight)
            }

        return trained_model, loss_per_epoch, eval_summary, entity_to_embedding, relation_to_embedding, params

    def _get_train_and_test_triples(self):

        pos_triples = _load_data(self.config[TRAINING_SET_PATH])

        if TEST_SET_PATH in self.config:
            train_pos = pos_triples
            test_pos = _load_data(path_to_data=self.config[TEST_SET_PATH])

        else:
            train_pos, test_pos = train_test_split(
                pos_triples,
                test_size=self.config[TEST_SET_RATIO],
                random_state=self.seed,
            )

        all_triples = np.concatenate([train_pos, test_pos], axis=0)
        self.entity_to_id, self.rel_to_id = create_mappings(triples=all_triples)
        mapped_pos_train_triples, _, _ = create_mapped_triples(
            triples=train_pos,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.rel_to_id,
        )

        mapped_pos_test_triples, _, _ = create_mapped_triples(
            triples=test_pos,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.rel_to_id,
        )

        return mapped_pos_train_triples, mapped_pos_test_triples

    def _get_train_triples(self):
        train_pos = _load_data(self.config[TRAINING_SET_PATH])

        self.entity_to_id, self.rel_to_id = create_mappings(triples=train_pos)

        mapped_pos_train_triples, _, _ = create_mapped_triples(
            triples=train_pos,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.rel_to_id,
        )

        return mapped_pos_train_triples


def _load_data(path_to_data: str):
    triples = np.loadtxt(
        fname=path_to_data,
        dtype=str,
        comments='@Comment@ Subject Predicate Object',
        delimiter='\t',
    )

    return triples
