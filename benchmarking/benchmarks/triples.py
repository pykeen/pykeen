"""Benchmarks for creating and splitting triples factories."""

import numpy as np

from pykeen.triples import TriplesFactory

from .common import SEED, make_factory


class TriplesFactorySuite:
    """Benchmark creating triples factories from labeled triples, and splitting them."""

    def setup(self) -> None:
        """Prepare labeled triples."""
        # larger than the default, since these operations are cheap
        self.factory = make_factory(num_entities=10_000, num_relations=100, num_triples=100_000)
        h, r, t = self.factory.mapped_triples.numpy().T
        self.labeled_triples = np.stack(
            [
                np.char.add("e", h.astype(str)),
                np.char.add("r", r.astype(str)),
                np.char.add("e", t.astype(str)),
            ],
            axis=-1,
        )

    def time_from_labeled_triples(self) -> None:
        """Time creating a triples factory from labeled triples."""
        TriplesFactory.from_labeled_triples(self.labeled_triples)

    def time_split(self) -> None:
        """Time splitting into training, validation and testing triples."""
        self.factory.split(ratios=[0.8, 0.1, 0.1], random_state=SEED)
