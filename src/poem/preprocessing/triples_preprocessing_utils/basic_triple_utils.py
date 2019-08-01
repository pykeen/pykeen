# -*- coding: utf-8 -*-

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from poem.utils import slice_triples

log = logging.getLogger(__name__)

def load_triples(path, delimiter='\t') -> np.array:
    """Load triples saved as tab separated values."""
    triples = np.loadtxt(
        fname=path,
        dtype=str,
        comments='@Comment@ Subject Predicate Object',
        delimiter=delimiter,
    )
    return triples


def create_entity_and_relation_mappings(triples: np.array) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Map entities and relations to ids."""
    # Slice triples returns two dimensional vectors that can't be used here
    subjects, relations, objects = triples[:, 0], triples[:, 1], triples[:, 2]

    # Sorting ensures consistent results when the triples are permuted
    entities = sorted(set(subjects).union(objects))
    relations = sorted(set(relations))

    entity_to_id: Dict[str, int] = {
        value: key
        for key, value in enumerate(entities)
    }

    rel_to_id: Dict[str, int] = {
        value: key
        for key, value in enumerate(relations)
    }

    return entity_to_id, rel_to_id


def create_triple_mappings(triples: np.array, are_triples_unique=True) -> Dict[tuple, int]:
    """Create mappings for triples."""
    if not are_triples_unique:
        triples = np.unique(ar=triples, axis=0)

    triples_to_id: Dict[tuple, int] = {
        tuple(value): key
        for key, value in enumerate(triples)
    }

    return triples_to_id


def map_triples_elements_to_ids(
        triples: np.array,
        entity_to_id: Optional[Dict[str, int]],
        rel_to_id: Optional[Dict[str, int]],
) -> np.ndarray:
    """Map entities and relations to predefined ids."""
    heads, relations, tails = slice_triples(triples)

    # When triples that don't exist are trying to be mapped, they get the id "-1"
    subject_column = np.vectorize(entity_to_id.get)(heads, [-1])
    relation_column = np.vectorize(rel_to_id.get)(relations, [-1])
    object_column = np.vectorize(entity_to_id.get)(tails, [-1])

    # Filter all non-existent triples
    subject_filter = subject_column < 0
    relation_filter = relation_column < 0
    object_filter = object_column < 0
    num_no_subject = subject_filter.sum()
    num_no_relation = relation_filter.sum()
    num_no_object = object_filter.sum()

    if (num_no_subject > 0) or (num_no_relation > 0) or (num_no_object > 0):
        log.warning(
            "You're trying to map triples with entities and/or relations that are not in the training set."
            "These triples will be excluded from the mapping")
        non_mappable_triples = (subject_filter | relation_filter | object_filter)
        subject_column = subject_column[~non_mappable_triples, None]
        relation_column = relation_column[~non_mappable_triples, None]
        object_column = object_column[~non_mappable_triples, None]
        log.warning(f"In total {non_mappable_triples.sum():.0f} from {triples.shape[0]:.0f} triples were filtered out")

    triples_of_ids = np.concatenate([subject_column, relation_column, object_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    # Note: Unique changes the order of the triples
    # Note: Using unique means implicit balancing of training samples
    return np.unique(ar=triples_of_ids, axis=0)


def get_unique_entity_pairs(triples, return_indices=False) -> np.array:
    """Extract all unique entity pairs from the triples."""
    heads, _, tails = slice_triples(triples)
    entity_pairs = np.concatenate([heads, tails], axis=1)
    return get_unique_pairs(pairs=entity_pairs, return_indices=return_indices)


def get_unique_subject_relation_pairs(triples, return_indices=False) -> np.ndarray:
    """Extract all unique subject relation pairs from the triples."""
    heads, relations, _ = slice_triples(triples)
    subject_relation_pairs = np.concatenate([heads, relations], axis=1)
    return get_unique_pairs(pairs=subject_relation_pairs, return_indices=return_indices)


def get_unique_pairs(pairs, return_indices=False) -> np.array:
    """Extract unique pairs."""
    # idx: Indices in triples of unique pairs
    _, idx = np.unique(pairs, return_index=True, axis=0)
    sorted_indices = np.sort(idx)
    # unique pairs where original order of triples is preserved
    unique_pairs = pairs[sorted_indices]

    if return_indices:
        return unique_pairs, sorted_indices
    return unique_pairs
