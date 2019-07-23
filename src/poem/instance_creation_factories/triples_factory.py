# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import numpy as np

from .instances import CWAInstances, OWAInstances
from ..preprocessing.instance_creation_utils.utils import create_multi_label_objects_instance
from ..preprocessing.triples_preprocessing_utils.basic_triple_utils import (
    create_entity_and_relation_mappings,
    load_triples,
    map_triples_elements_to_ids,
)


class TriplesFactory:
    """Create instances given the path to triples"""

    def __init__(self, path_to_triples: str) -> None:
        self.path_to_triples = path_to_triples
        # TODO: Check if lazy evaluation would make sense
        self.triples = load_triples(self.path_to_triples)
        self.entity_to_id, self.relation_to_id = create_entity_and_relation_mappings(self.triples)
        self.all_entities = np.array(list(self.entity_to_id.values()))
        self.num_entities = len(self.entity_to_id)
        self.num_relations = len(self.relation_to_id)

    def create_owa_instances(self) -> OWAInstances:
        mapped_triples = map_triples_elements_to_ids(
            triples=self.triples,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.relation_to_id,
        )
        return OWAInstances(
            instances=mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    def create_cwa_instances(self):
        mapped_triples = map_triples_elements_to_ids(
            triples=self.triples,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.relation_to_id,
        )

        s_r_to_mulit_objects = create_multi_label_objects_instance(
            triples=mapped_triples,
        )

        subject_relation_pairs = np.array(list(s_r_to_mulit_objects.keys()), dtype=np.float)
        labels = list(s_r_to_mulit_objects.values())

        return CWAInstances(
            instances=subject_relation_pairs,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            labels=labels,
        )

    def map_triples_to_id(self, path_to_triples: str) -> np.array:
        """Load triples and map to ids based on the existing id mappings of the triples factory"""
        triples = load_triples(path_to_triples)
        mapped_triples = map_triples_elements_to_ids(
            triples=triples,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.relation_to_id,
        )
        return mapped_triples
