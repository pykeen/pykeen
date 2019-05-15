# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np

from ...utils import slice_triples


def create_multi_label_relation_instances(unique_entity_pairs: np.array,
                                          triples: np.array,
                                          num_relations: int,
                                          create_class_other=False
                                          ) -> Dict[tuple, np.array]:
    """Create for each (s,o) pair the multi relation label."""

    subjects, relations, objects = slice_triples(triples)
    entity_pairs = np.concatenate([subjects, objects], axis=1)

    # Create class 'other' for relations not contained in the KG
    if create_class_other:
        num_relations += 1

    s_t_to_multi_relations = create_multi_label_instances(unique_pairs=unique_entity_pairs,
                                                          pairs=entity_pairs,
                                                          elements=relations,
                                                          num_elements=num_relations)
    return s_t_to_multi_relations


def create_multi_label_objects_instance(unique_s_r_pairs: np.array,
                                        triples: np.array,
                                        num_entities: int) -> Dict[tuple, np.array]:
    """Create for each (s,r) pair the multi object label."""

    subjects, relations, objects = slice_triples(triples)
    s_r_pairs = np.concatenate([subjects, relations], axis=1)

    s_r_to_mulit_objects = create_multi_label_instances(unique_pairs=unique_s_r_pairs,
                                                        pairs=s_r_pairs,
                                                        elements=objects,
                                                        num_elements=num_entities)
    return s_r_to_mulit_objects


def create_multi_label_instances(unique_pairs: np.array,
                                 pairs: np.array,
                                 elements: np.array,
                                 num_elements) -> Dict[tuple, np.array]:
    """Create for each (element_1, element_2) pair the multi-label."""

    instance_to_multi_label = {}

    for unique_pair in unique_pairs:
        # Step 1: Get all corresponding elements of pair
        indices = np.where((pairs == unique_pair).all(-1))
        all_corresponding_elements_of_pair = np.array(np.sort(elements[indices]).tolist(), dtype=np.int)
        # Step 2: Create hot encoding labels
        multi_label = np.zeros(num_elements)
        np.put(multi_label, all_corresponding_elements_of_pair, np.ones(len(all_corresponding_elements_of_pair)))
        # Step 3: Save in dict
        instance_to_multi_label[tuple(unique_pair)] = multi_label

    return instance_to_multi_label
