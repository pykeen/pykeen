"""Shared data for the benchmarks.

Benchmarks whose cost depends on the graph structure, e.g., on the number of triples per (head, relation) pair, use
Kinships, the largest dataset shipped with PyKEEN. Benchmarks whose cost mainly grows with the number of entities, such
as scoring against all entities and evaluation, use a larger synthetic graph, since Kinships only has 104 entities.
"""

import torch

from pykeen.datasets import Kinships
from pykeen.triples import CoreTriplesFactory
from pykeen.triples.generation import generate_triples_factory

__all__ = [
    "EMBEDDING_DIM",
    "SEED",
    "load_kinships",
    "make_factory",
]

#: the embedding dimension of the benchmarked models
EMBEDDING_DIM = 64
#: the random seed
SEED = 42


def load_kinships() -> Kinships:
    """Load the Kinships dataset, which is shipped with PyKEEN, and use a single thread for more stable timings."""
    torch.set_num_threads(1)
    return Kinships()


def make_factory(
    num_entities: int = 1_000,
    num_relations: int = 20,
    num_triples: int = 10_000,
) -> CoreTriplesFactory:
    """Generate a synthetic triples factory, and use a single thread for more stable timings."""
    torch.set_num_threads(1)
    return generate_triples_factory(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        random_state=SEED,
    )
