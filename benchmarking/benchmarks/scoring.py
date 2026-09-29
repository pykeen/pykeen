"""Benchmarks for scoring with models."""

import torch

from pykeen.models import model_resolver
from pykeen.triples import KGInfo

from .common import EMBEDDING_DIM, SEED

#: the number of entities; larger than for training, since scoring is cheaper
NUM_ENTITIES = 5_000
#: the number of relations
NUM_RELATIONS = 100
#: the batch size
BATCH_SIZE = 256


class ScoringSuite:
    """Benchmark scoring triples and all tails with different interactions."""

    param_names = ["model"]
    params = [["DistMult", "TransE", "ComplEx", "RotatE"]]

    def setup(self, model: str) -> None:
        """Prepare the model and batch."""
        torch.set_num_threads(1)
        self.model = model_resolver.make(
            model,
            triples_factory=KGInfo(num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS),
            embedding_dim=EMBEDDING_DIM,
            random_seed=SEED,
        ).eval()
        generator = torch.Generator().manual_seed(SEED)
        self.hrt_batch = torch.stack(
            [
                torch.randint(NUM_ENTITIES, size=(BATCH_SIZE,), generator=generator),
                torch.randint(NUM_RELATIONS, size=(BATCH_SIZE,), generator=generator),
                torch.randint(NUM_ENTITIES, size=(BATCH_SIZE,), generator=generator),
            ],
            dim=-1,
        )

    def time_score_hrt(self, model: str) -> None:
        """Time scoring individual triples."""
        with torch.inference_mode():
            self.model.score_hrt(self.hrt_batch)

    def time_score_t(self, model: str) -> None:
        """Time scoring all tail entities."""
        with torch.inference_mode():
            self.model.score_t(self.hrt_batch[:, :2])
