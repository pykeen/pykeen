"""Non-parametric baselines."""

import itertools as itt
import time
from abc import ABC
from typing import Any, List, Mapping, Optional, Tuple, Type, cast

import click
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import scipy.sparse
import seaborn as sns
import torch
from more_click import verbose_option
from sklearn.preprocessing import normalize as sklearn_normalize
from tabulate import tabulate
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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
    row_index: int,
    normalize: bool = False,
) -> scipy.sparse.csr_matrix:
    """Create a co-occurrence matrix from triples."""
    assert row_index != col_index
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

    def collect_regularization_term(self) -> torch.FloatTensor:  # noqa:D102
        # TODO: this is not needed for non-parametric models!
        raise NotImplementedError

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        # TODO: this is not needed for evaluation
        raise NotImplementedError

    def score_r(self, ht_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        # TODO: this is not needed for evaluation
        raise NotImplementedError


class PseudoTypeBaseline(EvaluationOnlyModel):
    """Score based on entity-relation co-occurrence."""

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        normalize: bool = False,
    ):
        super().__init__(triples_factory=triples_factory, random_seed=0, preferred_device='cpu')
        self.head_per_relation = _get_csr_matrix(
            triples_factory=triples_factory, row_index=1, col_index=0, normalize=normalize,
        )
        self.tail_per_relation = _get_csr_matrix(
            triples_factory=triples_factory, row_index=1, col_index=2, normalize=normalize,
        )
        self.normalize = normalize

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        r = hr_batch[:, 1].cpu().numpy()
        return torch.from_numpy(self.tail_per_relation[r].todense())

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        r = rt_batch[:, 0].cpu().numpy()
        return torch.from_numpy(self.head_per_relation[r].todense())


class EntityCoOccurrenceBaseline(EvaluationOnlyModel):
    """Score based on entity-entity co-occurrence."""

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        normalize: bool = False,
    ):
        super().__init__(triples_factory=triples_factory, random_seed=0, preferred_device='cpu')
        self.head_per_tail = _get_csr_matrix(
            triples_factory=triples_factory, row_index=2, col_index=0, normalize=normalize,
        )
        self.tail_per_head = _get_csr_matrix(
            triples_factory=triples_factory, row_index=0, col_index=2, normalize=normalize,
        )
        self.normalize = normalize

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        h = hr_batch[:, 0].cpu().numpy()
        return torch.from_numpy(self.tail_per_head[h].todense())

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        t = rt_batch[:, 1].cpu().numpy()
        return torch.from_numpy(self.head_per_tail[t].todense())


def _get_relation_similarity(
    triples_factory: CoreTriplesFactory,
    to_inverse: bool = False,
    threshold: Optional[float] = None,
) -> scipy.sparse.csr_matrix:
    # TODO: overlap with inverse triple detection
    assert triples_factory.num_entities * triples_factory.num_relations < numpy.iinfo(int_type=int).max
    mapped_triples = numpy.asarray(triples_factory.mapped_triples)
    r = scipy.sparse.coo_matrix(
        (
            numpy.ones((mapped_triples.shape[0],), dtype=int),
            (
                mapped_triples[:, 1],
                triples_factory.num_entities * mapped_triples[:, 0] + mapped_triples[:, 2],
            ),
        ),
        shape=(triples_factory.num_relations, triples_factory.num_entities ** 2),
    )
    cardinality = numpy.asarray(r.sum(axis=1)).squeeze(axis=-1)
    if to_inverse:
        r2 = scipy.sparse.coo_matrix(
            (
                numpy.ones((mapped_triples.shape[0],), dtype=int),
                (
                    mapped_triples[:, 1],
                    triples_factory.num_entities * mapped_triples[:, 2] + mapped_triples[:, 0],
                ),
            ),
            shape=(triples_factory.num_relations, triples_factory.num_entities ** 2),
        )
    else:
        r2 = r
    intersection = numpy.asarray((r @ r2.T).todense())
    union = cardinality[:, None] + cardinality[None, :] - intersection
    sim = intersection.astype(numpy.float32) / union.astype(numpy.float32)
    if threshold is not None:
        sim[sim < threshold] = 0.0
    sim = scipy.sparse.csr_matrix(sim)
    sim.eliminate_zeros()
    return sim


class SoftInverseTripleBaseline(EvaluationOnlyModel):
    """Score based on relation similarity."""

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        threshold: Optional[float] = None,
    ):
        super().__init__(triples_factory=triples_factory, random_seed=0, preferred_device='cpu')
        # compute relation similarity matrix
        self.sim = _get_relation_similarity(triples_factory, to_inverse=False, threshold=threshold)
        self.sim_inv = _get_relation_similarity(triples_factory, to_inverse=True, threshold=threshold)

        mapped_triples = numpy.asarray(triples_factory.mapped_triples)
        self.rel_to_head = scipy.sparse.coo_matrix(
            (
                numpy.ones(shape=(triples_factory.num_triples,), dtype=numpy.float32),
                (mapped_triples[:, 1], mapped_triples[:, 0]),
            ),
            shape=(triples_factory.num_relations, triples_factory.num_entities),
        ).tocsr()
        self.rel_to_tail = scipy.sparse.coo_matrix(
            (
                numpy.ones(shape=(triples_factory.num_triples,), dtype=numpy.float32),
                (mapped_triples[:, 1], mapped_triples[:, 2]),
            ),
            shape=(triples_factory.num_relations, triples_factory.num_entities),
        ).tocsr()

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        r = hr_batch[:, 1]
        scores = self.sim[r, :] @ self.rel_to_tail + self.sim_inv[r, :] @ self.rel_to_head
        scores = numpy.asarray(scores.todense())
        return torch.from_numpy(scores)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        r = rt_batch[:, 0]
        scores = self.sim[r, :] @ self.rel_to_head + self.sim_inv[r, :] @ self.rel_to_tail
        scores = numpy.asarray(scores.todense())
        return torch.from_numpy(scores)


BENCHMARK_PATH = PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark.tsv')
METRICS = ['mrr', 'hits@1', 'hits@10', 'aamr', 'aamri']


@click.command()
@verbose_option
@click.option('--batch-size', default=1024, show_default=True)
@click.option('--rebuild', is_flag=True)
def main(batch_size: int, rebuild: bool):
    """Show-case baseline."""
    if not BENCHMARK_PATH.is_file() or rebuild:
        with logging_redirect_tqdm():
            df = _build(batch_size=batch_size)
    else:
        df = pd.read_csv(BENCHMARK_PATH, sep='\t')

    _plot(df)


def _plot(df: pd.DataFrame):
    keep = [col for col in df.columns if col not in METRICS]
    tsdf = pd.melt(
        df[[*keep, *METRICS]],
        id_vars=keep,
        value_vars=METRICS,
        var_name='metric',
    )

    # Plot relation between # triples and time, stratified by model
    # Interpretation: exponential relationship between # triples and time
    fig, ax = plt.subplots()
    sns.scatterplot(data=tsdf, x='triples', y='time', hue='model', ax=ax)
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.savefig(PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_timeplot.svg'))
    fig.savefig(PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_timeplot.png'), dpi=300)
    plt.close(fig)

    # Make a scatterplot grid showing relation between # triples and result, stratified by model and metric.
    # Interpretation: no dataset size dependence
    g = sns.FacetGrid(
        tsdf[~tsdf.metric.isin({'aamr', 'aamri'})],
        col="model",
        hue='dataset',
        row='metric',
        margin_titles=True,
    )
    g.map(sns.scatterplot, "triples", "value")
    g.add_legend()
    g.set(xscale='log')
    g.savefig(PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_scatterplot.svg'))
    g.savefig(PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_scatterplot.png'), dpi=300)
    plt.close(g.fig)


def _build(batch_size: int) -> pd.DataFrame:
    datasets = sorted(dataset_resolver, key=Dataset._sort_key)
    # CoDEx Large is the first dataset where this gets a bit out of hand
    datasets = datasets[:datasets.index(dataset_resolver.lookup('CoDExLarge'))]
    models_kwargs: List[Tuple[Type[EvaluationOnlyModel], Mapping[str, Any]]] = [
        (PseudoTypeBaseline, dict(normalize=True)),
        (EntityCoOccurrenceBaseline, dict(normalize=True)),
        (SoftInverseTripleBaseline, dict(threshold=0.97)),
    ]
    kwargs_keys = sorted({k for _, d in models_kwargs for k in d})

    records = []
    it = tqdm(itt.product(datasets, models_kwargs), desc='Baseline', total=len(datasets) * len(models_kwargs))
    for dataset_cls, (model_cls, kwargs) in it:
        model_name = model_cls.__name__[:-len('Baseline')]
        it.set_postfix({'dataset': dataset_cls.__name__, 'model': model_name})
        dataset = dataset_cls()
        model = model_cls(triples_factory=dataset.training, **kwargs)

        start_time = time.time()
        result = _evaluate_baseline(dataset, model, batch_size=batch_size)
        elapsed_seconds = time.time() - start_time

        records.append((
            dataset_cls.__name__,
            dataset.training.num_entities,
            dataset.training.num_relations,
            dataset.training.num_triples,
            model_name,
            *(kwargs.get(key) for key in kwargs_keys),
            elapsed_seconds,
            *(result.get_metric(metric) for metric in METRICS),
        ))

    columns = ['dataset', 'E', 'R', 'triples', 'model', *kwargs_keys, 'time', *METRICS]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(BENCHMARK_PATH, sep='\t', index=False)
    print(tabulate(df.round(3).values, headers=columns, tablefmt='github'))
    return df


def _evaluate_baseline(dataset: Dataset, model: Model, batch_size=None) -> RankBasedMetricResults:
    assert dataset.validation is not None
    evaluator = RankBasedEvaluator()
    return cast(RankBasedMetricResults, evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        evaluators=evaluator,
        batch_size=batch_size,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
        use_tqdm=100_000 < dataset.training.num_triples,  # only use for big datasets
    ))


if __name__ == '__main__':
    main()
