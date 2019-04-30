# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from typing import Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split

from kupp.triples_preprocessing_utils.basic_triple_utils import load_triples, create_entity_and_relation_mappings, \
    map_triples_elements_to_ids
from poem.constants import TRAINING_SET_PATH, TEST_SET_PATH, TEST_SET_RATIO, OWA, SEED, KG_ASSUMPTION
from poem.instance_creation_factories.instances import Instances, OWAInstances


class TriplesFactory:
    """."""

    def __init__(self, config):
        """."""
        self.config = config
        self.entity_to_id = None
        self.relation_to_id = None
        self.train_triples = None
        self.test_triples = None
        self.all_triples = None

    def map_triples(
        self, 
        train_triples, 
        test_triples=None, 
        validation_triples=None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """."""
        all_triples: np.ndarray = np.concatenate([train_triples, test_triples], axis=0)
        # Map each entity/relation to a unique id
        self.entity_to_id, self.relation_to_id = create_entity_and_relation_mappings(triples=all_triples)

        # Map triple elements to their ids
        mapped_train_triples = map_triples_elements_to_ids(
            triples=train_triples,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.relation_to_id,
        )

        if test_triples is None:
            return mapped_train_triples, None

        mapped_test_triples = map_triples_elements_to_ids(
            triples=test_triples,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.relation_to_id,
        )

        return mapped_train_triples, mapped_test_triples

    def get_test_triples(self, train_triples):
        """."""
        if TEST_SET_PATH in self.config:
            test_triples = load_triples(self.config[TEST_SET_PATH])
        else:
            train_triples, test_triples = train_test_split(
                train_triples,
                test_size=self.config.get(TEST_SET_RATIO, 0.1),
                random_state=self.config[SEED],
            )

        return train_triples, test_triples

    @property
    def is_OWA(self):
        """."""
        return self.config[KG_ASSUMPTION] == OWA

    def _create_cwa_instances(self):
        pass

    def _create_owa_instance_object(self, instances, entity_to_id, relation_to_id) -> Instances:
        """"""
        return OWAInstances(instances=instances,
                            entity_to_id=entity_to_id,
                            relation_to_id=relation_to_id)

    def _create_owa_train_and_test_instances(self, training_triples, test_triples):
        """."""
        training_instances = self._create_owa_instance_object(instances=training_triples,
                                                              entity_to_id=self.entity_to_id,
                                                              relation_to_id=self.relation_to_id)

        test_instances = self._create_owa_instance_object(instances=test_triples,
                                                          entity_to_id=self.entity_to_id,
                                                          relation_to_id=self.relation_to_id)

        return training_instances, test_instances

    def create_train_and_test_instances(self) -> (Instances, Instances):
        """."""

        # Step 1: Load training triples
        training_triples = load_triples(path=self.config[TRAINING_SET_PATH], delimiter='\t')
        self.train_triples = training_triples
        self.all_triples = training_triples

        # Step 2: Create test triples if requested
        train_triples, test_triples = self.get_test_triples(train_triples=training_triples)
        self.test_triples = test_triples
        self.all_triples = np.concatenate([train_triples, test_triples], axis=0)

        training_triples, test_triples = self.map_triples(train_triples=self.train_triples,
                                                          test_triples=self.test_triples)

        if self.is_OWA:
            return self._create_owa_train_and_test_instances(training_triples=training_triples,
                                                             test_triples=test_triples)
        else:
            # TODO: ADD CWA logic
            pass

    def create_instances(self) -> Instances:
        """"""

        # Step 1: Load training triples
        training_triples = load_triples(path=self.config[TRAINING_SET_PATH], delimiter='\t')
        self.train_triples = training_triples
        self.all_triples = training_triples

        training_triples, _ = self.map_triples(train_triples=self.train_triples,
                                               test_triples=self.test_triples)

        if self.is_OWA:
            return self._create_owa_instance_object(instances=training_triples,
                                                    entity_to_id=self.entity_to_id,
                                                    relation_to_id=self.relation_to_id)

        else:
            # TODO: ADD CWA logic
            pass





