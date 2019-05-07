# -*- coding: utf-8 -*-


from typing import Dict, Optional, Tuple

import numpy as np


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
    entities = np.unique(np.ndarray.flatten(np.concatenate([triples[:, 0:1], triples[:, 2:3]])))
    relations = np.unique(np.ndarray.flatten(triples[:, 1:2]).tolist())

    entity_to_id: Dict[int, str] = {
        value: key
        for key, value in enumerate(entities)
    }

    rel_to_id: Dict[int, str] = {
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


def map_triples_elements_to_ids(triples: np.array,
                                entity_to_id: Optional[Dict[int, str]],
                                rel_to_id: Optional[Dict[int, str]]) -> np.ndarray:
    """Map entities and relations to predefined ids."""

    subject_column = np.vectorize(entity_to_id.get)(triples[:, 0:1])
    relation_column = np.vectorize(rel_to_id.get)(triples[:, 1:2])
    object_column = np.vectorize(entity_to_id.get)(triples[:, 2:3])
    triples_of_ids = np.concatenate([subject_column, relation_column, object_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    # Note: Unique changes the order of the triples
    return np.unique(ar=triples_of_ids, axis=0)


def get_unique_entity_pairs(triples, return_indices=False) -> np.array:
    """Extract all unique entity pairs from the triples."""

    subjects, _, objects = slice_triples(triples)

    entity_pairs = np.concatenate([subjects, objects], axis=1)

    return get_unique_pairs(pairs=entity_pairs, return_indices=return_indices)


def get_unique_subject_relation_pairs(triples, return_indices=False) -> np.array:
    """Extract all unique subject relation pairs from the triples."""

    subjects, relations, _ = slice_triples(triples)

    subject_relation_pairs = np.concatenate([subjects, relations], axis=1)

    return get_unique_pairs(pairs=subject_relation_pairs, return_indices=return_indices)


def get_unique_pairs(pairs, return_indices=False) -> np.array:
    """Extract unique pairs."""

    # idx: Indices in triples of unique pairs
    _, idx = np.unique(pairs, return_index=True, axis=0)
    sorted_indices = np.sort(idx)
    # uniquoe pairs where original order of triples is preserved
    unique_pairs = pairs[sorted_indices]

    if return_indices:
        return unique_pairs, sorted_indices
    return unique_pairs


def slice_triples(triples):
    """Get the heads, relations, and tails from a matrix of triples."""

    heads = triples[:, 0:1]
    relations = triples[:, 1:2]
    triples = triples[:, 2:3]

    return heads, relations, triples
