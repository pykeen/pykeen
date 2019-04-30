# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict
from kupp.triples_preprocessing_utils.basic_triple_utils import load_triples, create_entity_and_relation_mappings, \
    map_triples_elements_to_ids
from poem.basic_utils import is_evaluation_requested
from poem.constants import TRAINING_SET_PATH, TEST_SET_PATH, TEST_SET_RATIO, OWA, CWA
from typing import Tuple, Dict


@dataclass
class Instances:
    """."""
    training_instances: np.ndarray

    test_instances = None
    has_test_instances = False

    validation_instances = None
    has_validation_instances = False

    multimodal_data: Dict[str, np.ndarray]
    has_multimodal_data = False

    entity_to_id: Dict[str, np.ndarray]
    relation_to_id: Dict[str, np.ndarray]

    kg_assumption: str


@dataclass
class CWAInstances(Instances):
    """."""
    training_labels: np.array
    test_labels: np.array
    kg_assumption = CWA


class TriplesFactory(object):
    """."""

    def __int__(self, config):
        """."""
        print("Hi")
        self.config = config
        self.entity_to_id = None
        self.relation_to_id = None
        self.train_triples = None
        self.test_triples = None
        self.all_triples = None

    def map_triples(self, train_triples, test_triples=None, validation_triples = None) -> Tuple[np.ndarray, np.ndarray]:
        """."""
        all_triples: np.ndarray = np.concatenate([train_triples, test_triples], axis=0)
        # Map each entity/relation to a unique id
        self.entity_to_id, self.relation_to_id = create_entity_and_relation_mappings(triples=all_triples)

        # Map triple elements to their ids
        mapped_train_triples = map_triples_elements_to_ids(triples=train_triples,
                                                           entity_to_id=self.entity_to_id,
                                                           rel_to_id=self.relation_to_id)

        if test_triples is None:
            return mapped_train_triples, None

        mapped_test_triples = map_triples_elements_to_ids(triples=test_triples,
                                                          entity_to_id=self.entity_to_id,
                                                          rel_to_id=self.relation_to_id)

        return mapped_train_triples, mapped_test_triples

    def get_test_triples(self, train_triples):
        """."""
        if TEST_SET_PATH in self.config:
            test_triples = load_triples(self.config[TEST_SET_PATH])
        else:
            train_triples, test_triples = train_test_split(
                train_triples,
                test_size=self.config.get(TEST_SET_RATIO, 0.1),
                random_state=self.seed,
            )

        return train_triples, test_triples

    def create_instances(self) -> Instances:
        """"""

        # Step 1: Load training triples
        training_triples = load_triples(path=self.config[TRAINING_SET_PATH], delimiter='\t')
        self.train_triples = training_triples
        self.all_triples = training_triples

        # Step 2: Create test triples if requested
        if is_evaluation_requested:
            train_triples, test_triples = self.get_test_triples(train_triples=training_triples)
            self.test_triples = test_triples
            self.all_triples = np.concatenate([train_triples, test_triples], axis=0)

        training_triples, test_triples = self.map_triples(train_triples=self.train_triples,
                                                          test_triples=self.test_triples)

        return Instances(training_triples=training_triples,
                         test_triples=test_triples,
                         has_test_instances=True,
                         entity_to_id=self.entity_to_id,
                         relation_to_id=self.relation_to_id,
                         kg_assumption=OWA)


        # TODO
        #if is_cwa(self.config):
         #   # Create multi-labels
          #  pass

