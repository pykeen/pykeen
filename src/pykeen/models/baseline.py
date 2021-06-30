"""Non-parametric baselines."""

import itertools as itt
from abc import ABC
from typing import cast

import click
import numpy
import pandas as pd
import scipy.sparse
import torch
from more_click import verbose_option
from sklearn.preprocessing import normalize as sklearn_normalize
from tabulate import tabulate
from tqdm import tqdm

from pykeen.constants import PYKEEN_EXPERIMENTS
from pykeen.datasets import Dataset, dataset_resolver
from pykeen.evaluation import RankBasedEvaluator, RankBasedMetricResults, evaluate
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


BENCHMARK_PATH = PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark.tsv')


@click.command()
@verbose_option
def main():
    """Show-case baseline."""
    # datasets = sorted(dataset_resolver, key=Dataset._sort_key) # when it's all done
    # datasets = datasets[:3]  # for testing
    datasets = [dataset_resolver.lookup('fb15k237')]
    models = [
        PseudoTypeBaseline,
        EntityCoOccurrenceBaseline,
    ]

    records = {}
    it = tqdm(itt.product(datasets, models), desc='Baseline', total=len(datasets) * len(models))
    for dataset_cls, model_cls in it:
        it.set_postfix({'dataset': dataset_cls.__name__, 'model': model_cls.__name__})
        dataset = dataset_cls()
        model = model_cls(triples_factory=dataset.training, normalize=True)
        result = _evaluate_baseline(dataset, model)
        records[dataset_cls.__name__, model_cls.__name__] = _get_record(result)
    df = pd.DataFrame.from_dict(records, orient='index').reset_index()
    df.to_csv(BENCHMARK_PATH, sep='\t', index=False)
    print(tabulate(df.round(3), headers=['Index', *df.columns]))


def _evaluate_baseline(dataset: Dataset, model: Model) -> RankBasedMetricResults:
    evaluator = RankBasedEvaluator()
    return cast(RankBasedMetricResults, evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        evaluators=evaluator,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
        use_tqdm=100_000 < dataset.training.num_triples,  # only use for big datasets
    ))


def _get_record(m: RankBasedMetricResults) -> dict[str, float]:
    return {
        'mrr': m.get_metric('mrr'),
        'hits@1': m.get_metric('hits@1'),
        'hits@10': m.get_metric('hits@10'),
        'aamr': m.get_metric('aamr'),
        'aamri': m.get_metric('aamri'),
    }


if __name__ == '__main__':
    main()
