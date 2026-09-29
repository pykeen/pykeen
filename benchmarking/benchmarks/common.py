"""Shared synthetic data for the benchmarks."""

import torch

from pykeen.triples import CoreTriplesFactory
from pykeen.triples.generation import generate_triples_factory

__all__ = [
    "EMBEDDING_DIM",
    "SEED",
    "make_factory",
]

#: the embedding dimension of the benchmarked models
EMBEDDING_DIM = 64
#: the random seed
SEED = 42


def make_factory(
    num_entities: int = 1_000,
    num_relations: int = 20,
    num_triples: int = 10_000,
) -> CoreTriplesFactory:
    """Generate a synthetic triples factory, and use a single thread for more stable timings.

    The default size keeps an LCWA training epoch on a CPU in the order of seconds.
    """
    torch.set_num_threads(1)
    return generate_triples_factory(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        random_state=SEED,
    )
