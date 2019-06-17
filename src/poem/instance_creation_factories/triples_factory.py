# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

import numpy as np

from .instances import CWAInstances, OWAInstances
from ..preprocessing.instance_creation_utils.utils import create_multi_label_objects_instance
from ..preprocessing.triples_preprocessing_utils.basic_triple_utils import (
    get_unique_subject_relation_pairs, map_triples_elements_to_ids,
)


class TriplesFactory:
    def __init__(self, entity_to_id, relation_to_id):
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id

    def create_owa_instances(self, triples) -> OWAInstances:
        mapped_triples = map_triples_elements_to_ids(
            triples=triples,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.relation_to_id,
        )
        return OWAInstances(
            instances=mapped_triples,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    def create_cwa_instances(self, triples):
        mapped_triples = map_triples_elements_to_ids(
            triples=triples,
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
