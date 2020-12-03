# -*- coding: utf-8 -*-

"""Utilities for generating triples."""

from typing import Mapping
from uuid import uuid4

import numpy as np

from .triples_factory import TriplesFactory
from .utils import get_entities, get_relations
from ..typing import RandomHint
from ..utils import ensure_random_state

__all__ = [
    'generate_triples',
    'generate_labeled_triples',
    'generate_triples_factory',
]


def generate_triples(
    num_entities: int = 33,
    num_relations: int = 7,
    num_triples: int = 101,
    compact: bool = True,
    random_state: RandomHint = None,
) -> np.ndarray:
    """Generate random triples."""
    random_state = ensure_random_state(random_state)
    rv = np.stack([
        random_state.randint(num_entities, size=(num_triples,)),
        random_state.randint(num_relations, size=(num_triples,)),
        random_state.randint(num_entities, size=(num_triples,)),
    ], axis=1)

    if compact:
        new_entity_id = {
            entity: i
            for i, entity in enumerate(sorted(get_entities(rv)))
        }
        new_relation_id = {
            relation: i
            for i, relation in enumerate(sorted(get_relations(rv)))
        }
        rv = np.asarray([
            [new_entity_id[h], new_relation_id[r], new_entity_id[t]]
            for h, r, t in rv
        ], dtype=int)

    return rv


def generate_labeled_triples(
    num_entities: int = 33,
    num_relations: int = 7,
    num_triples: int = 101,
    random_state: RandomHint = None,
) -> np.ndarray:
    """Generate labeled random triples."""
    t = generate_triples(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        compact=False,
        random_state=random_state,
    )
    entity_id_to_label = _make_id_to_labels(num_entities)
    relation_id_to_label = _make_id_to_labels(num_relations)
    return np.asarray([
        (
            entity_id_to_label[h],
            relation_id_to_label[r],
            entity_id_to_label[t],
        )
        for h, r, t in t
    ], dtype=str)


def _make_id_to_labels(n: int) -> Mapping[int, str]:
    return {
        index: str(uuid4())
        for index in range(n)
    }


def generate_triples_factory(
    num_entities: int = 33,
    num_relations: int = 7,
    num_triples: int = 101,
    random_state: RandomHint = None,
    create_inverse_triples: bool = False,
) -> TriplesFactory:
    """Generate a triples factory with random triples."""
    triples = generate_labeled_triples(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        random_state=random_state,
    )
    return TriplesFactory.from_labeled_triples(
        triples=triples,
        create_inverse_triples=create_inverse_triples,
    )
