import random

import torch
import torch.optim as optim
import yaml

from utilities.constants import READER, KG_EMBEDDING_MODEL, NUM_ENTITIES, NUM_RELATIONS
from utilities.pipeline_helper import get_reader, get_kg_embedding_model, create_triples_and_mappings, \
    create_negative_triples


class Pipeline(object):

    def __init__(self, config):
        self.config = config
        self.corpus_reader = None
        self.kg_embedding_model = None
        self.eval_module = None
        # self._initialize_components(config=config)

    def _initialize_components(self, config):
        """

        :param config:
        :return:
        """
        # Initialize reader
        reader_config = config[READER]
        self.corpus_reader = get_reader(config=reader_config)

        # Initialize KG embedding model
        kb_embedding_model_config = config[KG_EMBEDDING_MODEL]
        self.kg_embedding_model = get_kg_embedding_model(config=kb_embedding_model_config)

    def start_pipeline(self, learning_rate, num_epochs, ratio_of_neg_triples, batch_size):
        """
        :return:
        """
        # Initialize reader
        reader_config = self.config[READER]
        self.corpus_reader = get_reader(config=reader_config)
        path_to_kg = self.corpus_reader.retreive_knowledge_graph()
        pos_tripels_of_ids, entity_to_id, rel_to_id = create_triples_and_mappings(path_to_kg=path_to_kg)

        # Initialize KG embedding model
        kb_embedding_model_config = self.config[KG_EMBEDDING_MODEL]
        kb_embedding_model_config[NUM_ENTITIES] = len(entity_to_id)
        kb_embedding_model_config[NUM_RELATIONS] = len(rel_to_id)
        self.kg_embedding_model = get_kg_embedding_model(config=kb_embedding_model_config)

        neg_triples = create_negative_triples(pos_triples=pos_tripels_of_ids,
                                              ratio_of_negative_triples=ratio_of_neg_triples)

        self._train(learning_rate, num_epochs, batch_size, pos_tripels_of_ids, neg_triples)

        return self.kg_embedding_model

    def _train(self, learning_rate, num_epochs, batch_size, pos_tripels, neg_triples):
        optimizer = optim.SGD(self.kg_embedding_model.parameters(), lr=learning_rate)

        total_loss = 0

        num_instances = max(len(pos_tripels), len(neg_triples))
        # num_batches = num_instances // num_epochs

        for epoch in range(num_epochs):
            for step in range(num_instances):
                pos_triple = torch.tensor(random.choice(pos_tripels), dtype=torch.long)
                neg_triple = torch.tensor(random.choice(neg_triples), dtype=torch.long)

                # Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                # model.zero_grad()
                # When to use model.zero_grad() and when optimizer.zero_grad() ?
                optimizer.zero_grad()

                loss = self.kg_embedding_model(pos_triple, neg_triple)

                loss.backward()
                optimizer.step()

                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()


