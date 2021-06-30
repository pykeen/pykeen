"""Non-parametric baselines."""
import pprint
from abc import ABC

import numpy
import scipy.sparse
import torch
from sklearn.preprocessing import normalize as sklearn_normalize

from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator, evaluate
from pykeen.models import Model
from pykeen.triples import CoreTriplesFactory


def _get_max_id(triples_factory: CoreTriplesFactory, index: int) -> int:
    """Get the number of entities or relations, depending on the selected column index."""
    return triples_factory.num_relations if index == 1 else triples_factory.num_entities


def _get_csr_matrix(
    triples_factory: CoreTriplesFactory,
    col_index: int,
    row_index: int = 1,
    normalize: bool = False,
) -> scipy.sparse.csr_matrix:
    """Create a co-occurrence matrix from triples."""
    row, col = triples_factory.mapped_triples.T[[row_index, col_index]]
    num_rows = _get_max_id(triples_factory=triples_factory, index=row_index)
    num_columns = _get_max_id(triples_factory=triples_factory, index=col_index)
    matrix = scipy.sparse.coo_matrix(
        (numpy.ones(triples_factory.num_triples), (row, col)),
        shape=(num_rows, num_columns),
    ).tocsr()
    if normalize:
        matrix = sklearn_normalize(matrix, norm="l1", axis=1)
    return matrix


class EvaluationOnlyModel(Model, ABC):
    """A model which only implements the methods used for evaluation."""

    def _reset_parameters_(self):
        # TODO: this is not needed for non-parametric models!
        raise NotImplementedError

    def collect_regularization_term(self) -> torch.FloatTensor:
        # TODO: this is not needed for non-parametric models!
        raise NotImplementedError

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:
        # TODO: this is not needed for evaluation
        raise NotImplementedError

    def score_r(self, ht_batch: torch.LongTensor) -> torch.FloatTensor:
        # TODO: this is not needed for evaluation
        raise NotImplementedError


class PseudoTypeBaseline(EvaluationOnlyModel):
    """Score based on entity-relation co-occurrence."""

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        normalize: bool = False,
    ):
        super().__init__(triples_factory=triples_factory)
        self.head_per_relation = _get_csr_matrix(triples_factory=triples_factory, col_index=0)
        self.tail_per_relation = _get_csr_matrix(triples_factory=triples_factory, col_index=2)
        self.normalize = normalize

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:
        r = hr_batch[:, 1].cpu().numpy()
        return torch.from_numpy(self.tail_per_relation[r].todense())

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:
        r = rt_batch[:, 0].cpu().numpy()
        return torch.from_numpy(self.head_per_relation[r].todense())


class EntityCoOccurrenceBaseline(EvaluationOnlyModel):
    """Score based on entity-entity co-occurrence."""

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        normalize: bool = False,
    ):
        super().__init__(triples_factory=triples_factory)
        self.head_per_tail = _get_csr_matrix(triples_factory=triples_factory, row_index=2, col_index=0)
        self.tail_per_head = _get_csr_matrix(triples_factory=triples_factory, row_index=0, col_index=2)
        self.normalize = normalize

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:
        h = hr_batch[:, 0].cpu().numpy()
        return torch.from_numpy(self.tail_per_head[h].todense())

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:
        t = rt_batch[:, 1].cpu().numpy()
        return torch.from_numpy(self.head_per_tail[t].todense())


def main():
    """Show-case baseline."""
    # achieves ~23% MRR / ~17% H@1 / ~35% H@10 in around ~1min total train+eval time on cpu
    dataset = get_dataset(dataset="fb15k237")
    model = PseudoTypeBaseline(triples_factory=dataset.training, normalize=True)
    evaluator = RankBasedEvaluator()
    result = evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        evaluators=evaluator,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ]
    )
    pprint.pprint(result.to_dict())


if __name__ == '__main__':
    main()
