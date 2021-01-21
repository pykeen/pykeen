# -*- coding: utf-8 -*-

"""Utilities for generating triples."""

import torch

from .triples_factory import CoreTriplesFactory
from .utils import get_entities, get_relations
from ..typing import TorchRandomHint
from ..utils import ensure_torch_random_state

__all__ = [
    'generate_triples',
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

    rv = torch.stack([
        torch.randint(num_entities, size=(num_triples,), generator=random_state),
        torch.randint(num_relations, size=(num_triples,), generator=random_state),
        torch.randint(num_entities, size=(num_triples,), generator=random_state),
    ], dim=1)

    if compact:
        new_entity_id = {
            entity: i
            for i, entity in enumerate(sorted(get_entities(rv)))
        }
        new_relation_id = {
            relation: i
            for i, relation in enumerate(sorted(get_relations(rv)))
        }
        rv = torch.as_tensor(
            data=[
                [new_entity_id[h], new_relation_id[r], new_entity_id[t]]
                for h, r, t in rv.tolist()
            ],
            dtype=torch.long,
        )

    return rv


def generate_triples_factory(
    num_entities: int = 33,
    num_relations: int = 7,
    num_triples: int = 101,
    random_state: TorchRandomHint = None,
    create_inverse_triples: bool = False,
) -> CoreTriplesFactory:
    """Generate a triples factory with random triples."""
    mapped_triples = generate_triples(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        random_state=random_state,
    )
    return CoreTriplesFactory.create(
        mapped_triples=mapped_triples,
        create_inverse_triples=create_inverse_triples,
    )
