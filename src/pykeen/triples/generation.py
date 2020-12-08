# -*- coding: utf-8 -*-

"""Utilities for generating triples."""

from typing import Mapping
from uuid import uuid4

import numpy as np
import torch

from .triples_factory import TriplesFactory
from .utils import get_entities, get_relations
from ..typing import TorchRandomHint
from ..utils import ensure_torch_random_state

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
    random_state: TorchRandomHint = None,
) -> torch.LongTensor:
    """Generate random triples in a torch tensor."""
    random_state = ensure_torch_random_state(random_state)

    mapped_triples = torch.stack([
        torch.randint(num_entities, size=(num_triples,), generator=random_state),
        torch.randint(num_relations, size=(num_triples,), generator=random_state),
        torch.randint(num_entities, size=(num_triples,), generator=random_state),
    ], dim=1)

    if compact:
        new_entity_id = {
            entity: i
            for i, entity in enumerate(sorted(get_entities(mapped_triples)))
        }
        new_relation_id = {
            relation: i
            for i, relation in enumerate(sorted(get_relations(mapped_triples)))
        }
        mapped_triples = torch.tensor([
            [
                new_entity_id[h],
                new_relation_id[r],
                new_entity_id[t],
            ]
            for h, r, t in mapped_triples.tolist()
        ], dtype=torch.long)

    return mapped_triples


def generate_labeled_triples(
    num_entities: int = 33,
    num_relations: int = 7,
    num_triples: int = 101,
    random_state: TorchRandomHint = None,
) -> np.ndarray:
    """Generate labeled random triples."""
    mapped_triples = generate_triples(
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
        for h, r, t in mapped_triples
    ], dtype=str)


def _make_id_to_labels(n: int) -> Mapping[int, str]:
    return {
        index: str(uuid4())
        for index in range(n)
    }


def _make_label_to_ids(n: int) -> Mapping[str, int]:
    return {v: k for k, v in _make_id_to_labels(n).items()}


def generate_triples_factory(
    num_entities: int = 33,
    num_relations: int = 7,
    num_triples: int = 101,
    random_state: TorchRandomHint = None,
    create_inverse_triples: bool = False,
) -> TriplesFactory:
    """Generate a triples factory with random triples."""
    mapped_triples = generate_triples(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        random_state=random_state,
    )
    return TriplesFactory(
        entity_to_id=_make_label_to_ids(num_entities),
        relation_to_id=_make_label_to_ids(num_relations),
        mapped_triples=mapped_triples,
        create_inverse_triples=create_inverse_triples,
    )
