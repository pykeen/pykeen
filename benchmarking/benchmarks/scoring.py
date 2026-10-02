"""Benchmarks for scoring with models."""

import torch

from pykeen.models import model_resolver, Model

from .common import EMBEDDING_DIM, SEED, make_factory

#: the number of entities; larger than for training, since scoring is cheaper
NUM_ENTITIES = 5_000
#: the number of relations
NUM_RELATIONS = 100
#: the number of triples; only used to determine the numbers of entities and relations
NUM_TRIPLES = 50_000
#: the batch size
BATCH_SIZE = 256


class ScoringSuite:
    """Benchmark scoring triples and all tails with different interactions."""

    param_names = ["model"]
    # one model per implementation pattern of the interaction function, rather than all models: product and sum
    # (DistMult), einsum (ComplEx), distance (TransE), complex rotation and distance (RotatE), and neural network
    # (ConvE). Code shared by all models is covered by any of them.
    params = [["DistMult", "TransE", "ComplEx", "RotatE", "ConvE"]]

    model: Model
    hrt_batch: torch.Tensor

    def setup(self, model: str) -> None:
        """Prepare the model and batch."""
        # note: a generated triples factory, rather than KGInfo, whose signature changed after v1.11.1
        factory = make_factory(num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS, num_triples=NUM_TRIPLES)
        self.model = model_resolver.make(
            model, triples_factory=factory, embedding_dim=EMBEDDING_DIM, random_seed=SEED
        ).eval()
        self.hrt_batch = factory.mapped_triples[:BATCH_SIZE]

    def time_score_hrt(self, model: str) -> None:
        """Time scoring individual triples."""
        with torch.inference_mode():
            self.model.score_hrt(self.hrt_batch)

    def time_score_t(self, model: str) -> None:
        """Time scoring all tail entities."""
        with torch.inference_mode():
            self.model.score_t(self.hrt_batch[:, :2])
