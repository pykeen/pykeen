"""Benchmarks for evaluation."""

from pykeen.datasets import UMLS
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import DistMult
from pykeen.triples import CoreTriplesFactory

from .common import EMBEDDING_DIM, SEED, make_factory

#: the number of evaluation triples of the synthetic graph
NUM_EVALUATION_TRIPLES = 1_000
#: the evaluation batch size
BATCH_SIZE = 256


class RankBasedEvaluationSuite:
    """Benchmark rank-based evaluation of head and tail prediction.

    The synthetic graph shows costs which grow with the number of entities. UMLS shows costs which depend on the
    number of filtered candidates per query: in the synthetic graph, almost all queries only have a single known
    answer, whereas in UMLS about 15% of the candidates are filtered on average, and all but one for some queries.
    """

    param_names = ["dataset", "filtered"]
    params = [["synthetic", "umls"], [False, True]]
    repeat = (1, 5, 60.0)
    number = 1

    def setup(self, dataset: str, filtered: bool) -> None:
        """Prepare the model and evaluation triples."""
        factory: CoreTriplesFactory
        if dataset == "umls":
            umls = UMLS()
            if umls.validation is None:
                raise RuntimeError
            factory = umls.training
            self.mapped_triples = umls.testing.mapped_triples
            filter_triples = [umls.training.mapped_triples, umls.validation.mapped_triples]
        else:
            factory = make_factory()
            self.mapped_triples = factory.mapped_triples[:NUM_EVALUATION_TRIPLES]
            filter_triples = [factory.mapped_triples]
        self.additional_filter_triples = filter_triples if filtered else None
        self.model = DistMult(triples_factory=factory, embedding_dim=EMBEDDING_DIM, random_seed=SEED)
        self.evaluator = RankBasedEvaluator(filtered=filtered)

    def time_evaluate(self, dataset: str, filtered: bool) -> None:
        """Time evaluation."""
        self.evaluator.evaluate(
            model=self.model,
            mapped_triples=self.mapped_triples,
            additional_filter_triples=self.additional_filter_triples,
            batch_size=BATCH_SIZE,
            use_tqdm=False,
        )
