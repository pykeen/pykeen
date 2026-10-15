"""Benchmarks for creating and splitting triples factories, on Kinships."""

import numpy as np

from pykeen.datasets.kinships import KINSHIPS_TEST_PATH, KINSHIPS_TRAIN_PATH, KINSHIPS_VALIDATE_PATH
from pykeen.triples import TriplesFactory
from pykeen.triples.utils import load_triples

from .common import SEED


class TriplesFactorySuite:
    """Benchmark creating triples factories from labeled triples, and splitting them."""

    labeled_triples: np.ndarray
    factory: TriplesFactory

    def setup(self) -> None:
        """Prepare the labeled triples of all splits."""
        self.labeled_triples = np.concatenate(
            [load_triples(path) for path in (KINSHIPS_TRAIN_PATH, KINSHIPS_VALIDATE_PATH, KINSHIPS_TEST_PATH)]
        )
        self.factory = TriplesFactory.from_labeled_triples(self.labeled_triples)

    def time_from_labeled_triples(self) -> None:
        """Time creating a triples factory from labeled triples."""
        TriplesFactory.from_labeled_triples(self.labeled_triples)

    def time_split(self) -> None:
        """Time splitting into training, validation and testing triples."""
        self.factory.split(ratios=[0.8, 0.1, 0.1], random_state=SEED)
