# -*- coding: utf-8 -*-
import logging
from collections import OrderedDict

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from utilities.constants import READER, KG_EMBEDDING_MODEL, NUM_ENTITIES, NUM_RELATIONS, EVALUATOR, PREFERRED_DEVICE, \
    GPU, HPO, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE
from utilities.hpo_initialization_utils import get_hyper_parameter_optimizer
from utilities.instance_creation_utils import create_mapped_triples, create_negative_triples
from utilities.module_initialization_utils import get_evaluator, get_reader, \
    get_kg_embedding_model
from utilities.train_utils import train

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
        reader_config = self.config[READER]
        self.corpus_reader = get_reader(config=reader_config)
        path_to_kg = self.corpus_reader.retreive_knowledge_graph()

        evaluator_config = self.config[EVALUATOR]
        evaluator = get_evaluator(config=evaluator_config)

        if is_hpo_mode:
            hp_optimizer_config = self.config[HPO]
            hp_optimizer = get_hyper_parameter_optimizer(hp_optimizer_config, evaluator)
            trained_model, train_entity_to_id, train_rel_to_id, eval_result, metric_string = hp_optimizer.optimize_hyperparams(
                self.config, path_to_kg, self.device, self.seed)
        else:
            train_params = self.config['data_params']
            ratio_test_data = train_params['ratio_test_data']

            pos_triples = np.loadtxt(fname=path_to_kg, dtype=str, comments='@Comment@ Subject Predicate Object')
            neg_triples = create_negative_triples(seed=self.seed, pos_triples=pos_triples)
            train_pos, test_pos, train_neg, test_neg = train_test_split(pos_triples, neg_triples,
                                                                        test_size=ratio_test_data,
                                                                        random_state=self.seed)

            mapped_pos_tripels, train_entity_to_id, train_rel_to_id = create_mapped_triples(pos_triples)
            mapped_neg_triples, _, _ = create_mapped_triples(pos_triples, entity_to_id=train_entity_to_id,
                                                             rel_to_id=train_rel_to_id)

            # Initialize KG embedding model
            kb_embedding_model_config = self.config[KG_EMBEDDING_MODEL]
            kb_embedding_model_config[NUM_ENTITIES] = len(train_entity_to_id)
            kb_embedding_model_config[NUM_RELATIONS] = len(train_rel_to_id)
            kg_embedding_model = get_kg_embedding_model(config=kb_embedding_model_config)

            batch_size = kb_embedding_model_config[BATCH_SIZE]
            num_epochs = kb_embedding_model_config[NUM_EPOCHS]
            lr = kb_embedding_model_config[LEARNING_RATE]

            learning_rate = kb_embedding_model_config[LEARNING_RATE]

            log.info("-------------Train KG Embeddings-------------")
            print(batch_size)
            trained_model = train(kg_embedding_model=kg_embedding_model, learning_rate=lr, num_epochs=num_epochs,
                                  batch_size=batch_size, pos_triples=mapped_pos_tripels, neg_triples=mapped_neg_triples,
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

        return trained_model, eval_summary, entity_to_embedding, relation_to_embedding
