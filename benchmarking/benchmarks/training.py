"""Benchmarks for training on Kinships."""

from collections.abc import Sequence
from typing import Any, ClassVar

from pykeen.datasets import Kinships
from pykeen.models import Model, model_resolver
from pykeen.training import TrainingLoop, training_loop_resolver
from pykeen.triples import TriplesFactory

from .common import EMBEDDING_DIM, SEED

#: the training batch size
BATCH_SIZE = 256


class TrainingSuite:
    """Benchmark a single training epoch."""

    param_names: ClassVar[Sequence[str]] = ["training_loop", "model"]
    params: ClassVar[Sequence[Sequence[Any]]] = [["slcwa", "lcwa"], ["DistMult", "RotatE"]]
    # an epoch takes up to a few seconds; limit the number of repetitions
    repeat: ClassVar[Sequence[int | float]] = (1, 5, 60.0)
    number: ClassVar[int] = 1

    factory: TriplesFactory
    model: Model
    training_loop: TrainingLoop

    def setup(self, training_loop: str, model: str) -> None:
        """Prepare the model and training loop."""
        self.factory = Kinships().training
        self.model = model_resolver.make(
            model, triples_factory=self.factory, embedding_dim=EMBEDDING_DIM, random_seed=SEED
        )
        self.training_loop = training_loop_resolver.make(training_loop, model=self.model, triples_factory=self.factory)

    def time_train_epoch(self, training_loop: str, model: str) -> None:
        """Time one training epoch."""
        self.training_loop.train(
            triples_factory=self.factory,
            num_epochs=1,
            batch_size=BATCH_SIZE,
            use_tqdm=False,
            use_tqdm_batch=False,
        )

    def peakmem_train_epoch(self, training_loop: str, model: str) -> None:
        """Measure the peak memory of one training epoch."""
        self.time_train_epoch(training_loop=training_loop, model=model)
