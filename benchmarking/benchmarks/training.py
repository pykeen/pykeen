"""Benchmarks for training."""

from pykeen.models import model_resolver
from pykeen.training import training_loop_resolver
from pykeen.triples import CoreTriplesFactory

from .common import EMBEDDING_DIM, SEED, make_factory

#: the training batch size
BATCH_SIZE = 1024


class _TrainingSuite:
    """Benchmark a single training epoch."""

    param_names = ["model"]
    # an epoch takes up to a few seconds; limit the number of repetitions
    repeat = (1, 5, 60.0)
    number = 1

    #: the training loop
    training_loop_name: str

    def make_factory(self) -> CoreTriplesFactory:
        """Create the training triples."""
        return make_factory()

    def setup(self, model: str) -> None:
        """Prepare the model and training loop."""
        self.factory = self.make_factory()
        self.model = model_resolver.make(
            model, triples_factory=self.factory, embedding_dim=EMBEDDING_DIM, random_seed=SEED
        )
        self.training_loop = training_loop_resolver.make(
            self.training_loop_name, model=self.model, triples_factory=self.factory
        )

    def time_train_epoch(self, model: str) -> None:
        """Time one training epoch."""
        self.training_loop.train(
            triples_factory=self.factory,
            num_epochs=1,
            batch_size=BATCH_SIZE,
            use_tqdm=False,
            use_tqdm_batch=False,
        )

    def peakmem_train_epoch(self, model: str) -> None:
        """Measure the peak memory of one training epoch."""
        self.time_train_epoch(model=model)


class SLCWATrainingSuite(_TrainingSuite):
    """Benchmark a single sLCWA training epoch."""

    params = [["DistMult", "RotatE"]]
    training_loop_name = "slcwa"


class LCWATrainingSuite(_TrainingSuite):
    """Benchmark a single LCWA training epoch.

    Scoring all tails for each (head, relation) pair is expensive on a CPU, so this suite uses a smaller graph, and
    ComplEx instead of RotatE, whose scoring against all entities is considerably slower.
    """

    params = [["DistMult", "ComplEx"]]
    training_loop_name = "lcwa"

    def make_factory(self) -> CoreTriplesFactory:  # noqa: D102
        return make_factory(num_triples=2_500)
