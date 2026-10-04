"""Shared data for the benchmarks.

Benchmarks whose cost depends on the graph structure, e.g., on the number of triples per (head, relation) pair, use
Kinships, the largest dataset shipped with PyKEEN. Benchmarks whose cost mainly grows with the number of entities, such
as scoring against all entities and evaluation, use a larger synthetic graph, since Kinships only has 104 entities.
Evaluation additionally uses UMLS, since the number of filtered candidates per query is much more skewed than for
Kinships, and almost always one for the synthetic graph.
"""

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
    """Generate a synthetic triples factory."""
    return generate_triples_factory(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        random_state=SEED,
    )
