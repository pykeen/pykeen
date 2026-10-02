"""Benchmarks for negative sampling on Kinships."""

from collections.abc import Sequence
from typing import Any, ClassVar

from pykeen.datasets import Kinships
from pykeen.sampling import negative_sampler_resolver

#: the batch size of positive triples
BATCH_SIZE = 1024
#: the number of negatives per positive triple
NUM_NEGS_PER_POS = 32


class NegativeSamplingSuite:
    """Benchmark sampling negatives for a batch, with different filterers."""

    param_names: ClassVar[Sequence[str]] = ["negative_sampler", "filterer"]
    params: ClassVar[Sequence[Sequence[Any]]] = [["basic", "bernoulli"], ["none", "python-set", "bloom"]]

    def setup(self, negative_sampler: str, filterer: str) -> None:
        """Prepare the negative sampler and positive batch."""
        factory = Kinships().training
        self.sampler = negative_sampler_resolver.make(
            negative_sampler,
            mapped_triples=factory.mapped_triples,
            num_entities=factory.num_entities,
            num_relations=factory.num_relations,
            num_negs_per_pos=NUM_NEGS_PER_POS,
            filterer=None if filterer == "none" else filterer,
        )
        self.positive_batch = factory.mapped_triples[:BATCH_SIZE]

    def time_sample(self, negative_sampler: str, filterer: str) -> None:
        """Time sampling negatives."""
        self.sampler.sample(positive_batch=self.positive_batch)
