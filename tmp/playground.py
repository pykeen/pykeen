# https://github.com/GTmac/HARP
from typing import Optional, Sequence
import torch
from pykeen.datasets import get_dataset
from pykeen.evaluation.rank_based_evaluator import RankBasedEvaluator
from pykeen.models.unimodal.distmult import DistMult
from pykeen.nn.emb import Embedding
from pykeen.pipeline.api import pipeline
from pykeen.training.slcwa import SLCWATrainingLoop
from pykeen.triples.triples_factory import CoreTriplesFactory


class Reduction:
    def __init__(
        self,
        num_entities: Optional[int] = None,
    ) -> None:
        self.num_entities = num_entities
        self.entity_map = None

    def fit(self, factory: CoreTriplesFactory) -> None:
        raise NotImplementedError

    def transform(self, factory: CoreTriplesFactory) -> CoreTriplesFactory:
        mapped_triples = factory.mapped_triples.clone()
        mapped_triples[:, [0, 2]] = self.entity_map[mapped_triples[:, [0, 2]]]
        mapped_triples = mapped_triples.unique(dim=0)
        return CoreTriplesFactory.create(
            mapped_triples=mapped_triples,
            num_entities=self.num_entities,
            num_relations=factory.num_relations,
            create_inverse_triples=factory.create_inverse_triples,
        )

    def fit_transform(self, factory: CoreTriplesFactory) -> CoreTriplesFactory:
        self.fit(factory=factory)
        return self.transform(factory)


class RandomReduction(Reduction):
    def __init__(self, num_entities: int) -> None:
        super().__init__(num_entities=num_entities)

    def fit(self, factory: CoreTriplesFactory) -> None:
        self.entity_map = torch.randint(self.num_entities, size=(factory.num_entities,))


class EdgeCollapseReduction(Reduction):
    def __init__(self, num_edges: int) -> None:
        super().__init__()
        self.num_edges = num_edges

    def fit(self, factory: CoreTriplesFactory) -> None:
        a, b = factory.mapped_triples[:, [0, 2]].unique(dim=0)[: self.num_edges].t()
        self.entity_map = torch.arange(factory.num_entities)
        self.entity_map[a] = b
        unique, self.entity_map = self.entity_map.unique(return_inverse=True)
        self.num_entities = unique.numel()


def harp(
    factory: CoreTriplesFactory,
    num_edges: Sequence[int] = None,
    num_entities: int = None,
):
    # reductions
    reductions = [None]
    factories = [factory]
    for ne in num_entities:
        reduction = RandomReduction(num_entities=ne)
        factory = reduction.fit_transform(factory=factory)
        print(factory)
        reductions.append(reduction)
        factories.append(factory)

    # train from small to big
    old_repr = None
    for factory, reduction in zip(reversed(factories), reversed(reductions)):
        print(factory)
        model = DistMult(triples_factory=factory)
        # pre-initialize from coarse
        if old_repr is not None:
            model.entity_embeddings._embeddings.weight.data = old_repr
        optimizer = torch.optim.SGD(params=model.get_grad_params(), lr=0.01)
        SLCWATrainingLoop(
            model=model,
            triples_factory=factory,
            optimizer=optimizer,
        ).train(triples_factory=factory, num_epochs=1)
        # print(evaluator.evaluate(model=model, mapped_triples=factory.mapped_triples))
        if reduction is not None:
            old_repr = model.entity_embeddings(reduction.entity_map)
    return model


ds = get_dataset(dataset="wn18rr")
harp(
    factory=ds.training,
    # num_edges=[100_000, 100_000, 30_000],
    num_entities=[16384, 4_096, 1_024, 256, 64],
)
