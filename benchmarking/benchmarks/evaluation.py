"""Benchmarks for evaluation."""

from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import DistMult

from .common import EMBEDDING_DIM, SEED, make_factory

#: the number of evaluation triples
NUM_EVALUATION_TRIPLES = 1_000
#: the evaluation batch size
BATCH_SIZE = 256


class RankBasedEvaluationSuite:
    """Benchmark rank-based evaluation of head and tail prediction."""

    param_names = ["filtered"]
    params = [[False, True]]
    repeat = (1, 5, 60.0)
    number = 1

    def setup(self, filtered: bool) -> None:
        """Prepare the model and evaluation triples."""
        self.factory = make_factory()
        self.model = DistMult(triples_factory=self.factory, embedding_dim=EMBEDDING_DIM, random_seed=SEED)
        self.mapped_triples = self.factory.mapped_triples[:NUM_EVALUATION_TRIPLES]
        self.evaluator = RankBasedEvaluator(filtered=filtered)

    def time_evaluate(self, filtered: bool) -> None:
        """Time evaluation."""
        self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.mapped_triples,
            additional_filter_triples=[self.factory.mapped_triples] if filtered else None,
            batch_size=BATCH_SIZE,
            use_tqdm=False,
        )
