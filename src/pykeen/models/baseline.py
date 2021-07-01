# -*- coding: utf-8 -*-

"""Non-parametric baselines."""

import itertools as itt
import time
from abc import ABC
from functools import partial
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast

import click
import numpy
import pandas as pd
import scipy.sparse
import torch
from more_click import verbose_option
from sklearn.preprocessing import normalize as sklearn_normalize
from tabulate import tabulate
from tqdm import trange
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.logging import logging_redirect_tqdm

from pykeen.constants import PYKEEN_EXPERIMENTS
from pykeen.datasets import Dataset, dataset_resolver
from pykeen.evaluation import RankBasedEvaluator, RankBasedMetricResults, evaluate
from pykeen.models import Model
from pykeen.triples import CoreTriplesFactory

__all__ = [
    'EvaluationOnlyModel',
    'MarginalDistributionBaseline',
    'SoftInverseTripleBaseline',
]

BENCHMARK_PATH = PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark.tsv')
TEST_BENCHMARK_PATH = PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_test.tsv')
KS = (1, 5, 10, 50, 100)
METRICS = ['mrr', 'iamr', 'igmr', *(f'hits@{k}' for k in KS), 'aamr', 'aamri']


def get_csr_matrix(
    row_indices: numpy.ndarray,
    col_indices: numpy.ndarray,
    shape: Tuple[int, int],
) -> scipy.sparse.csr_matrix:
    """Create a sparse matrix, for the given non-zero locations."""
    # create sparse matrix
    matrix = scipy.sparse.coo_matrix(
        (numpy.ones(row_indices.shape, dtype=numpy.float32), (row_indices, col_indices)),
        shape=shape,
    ).tocsr()
    # remove duplicates (in-place)
    matrix.sum_duplicates()
    # store logits for sparse multiplication by sparse addition
    matrix.data = numpy.log(matrix.data)
    return matrix


class EvaluationOnlyModel(Model, ABC):
    """A model which only implements the methods used for evaluation."""

    def __init__(self, triples_factory: CoreTriplesFactory):
        """Non-parametric models take a minimal set of arguments.

        :param triples_factory: The training triples factory is used to assign the number of entities, relations,
            and inverse condition in the non-parametric model.
        """
        super().__init__(
            triples_factory=triples_factory,
            # These operations are deterministic and a random feed can be fixed
            # just to avoid warnings
            random_seed=0,
            # These operations do not need to be performed on a GPU
            preferred_device='cpu',
        )

    def _reset_parameters_(self):
        """Non-parametric models do not implement :meth:`Model._reset_parameters_`."""

    def collect_regularization_term(self):  # noqa:D102
        """Non-parametric models do not implement :meth:`Model.collect_regularization_term`."""

    def score_hrt(self, hrt_batch: torch.LongTensor):  # noqa:D102
        """Non-parametric models do not implement :meth:`Model.score_hrt`."""

    def score_r(self, ht_batch: torch.LongTensor):  # noqa:D102
        """Non-parametric models do not implement :meth:`Model.score_r`."""


def _score(
    entity_relation_batch: torch.LongTensor,
    per_entity: Optional[scipy.sparse.csr_matrix],
    per_relation: Optional[scipy.sparse.csr_matrix],
    num_entities: int,
) -> torch.FloatTensor:
    """Shared code for computing entity scores from marginals."""
    batch_size = entity_relation_batch.shape[0]

    # base case
    if per_entity is None and per_relation is None:
        return torch.full(size=(batch_size, num_entities), fill_value=1 / num_entities)

    e, r = entity_relation_batch.cpu().numpy().T

    # create empty sparse matrix (i.e., filled by zeros) representation logits
    scores = scipy.sparse.csr_matrix((batch_size, num_entities), dtype=numpy.float32)

    # use per-entity marginal distribution
    if per_entity is not None:
        scores += per_entity[e]

    # use per-relation marginal distribution
    if per_relation is not None:
        scores += per_relation[r]

    # convert to probabilities
    scores.data = numpy.exp(scores.data)
    scores = sklearn_normalize(scores, norm="l1")

    # note: we need to work with dense arrays only to comply with returning torch tensors. Otherwise, we could
    # stay sparse here, with a potential of a huge memory benefit on large datasets!
    return torch.from_numpy(scores.todense())


class MarginalDistributionBaseline(EvaluationOnlyModel):
    r"""
    Score based on marginal distributions.

    To predict scores for the tails, we simplify

    .. math ::
        P(t | h, r) = P(t | h) * P(t | r)

    Depending on the settings, we either set P(t | *) = 1/n, or estimate them by counting occurrences in the training
    triples.

    .. note ::
        This model cannot make use of GPU acceleration, since internally it uses scipy's sparse matrices.
    """

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        entity_margin: bool = True,
        relation_margin: bool = True,
    ):
        """
        Initialize the model.

        :param triples_factory:
            The triples factory containing the training triples.
        """
        super().__init__(
            triples_factory=triples_factory,
            random_seed=0,  # TODO: Why do we provide the random seed?
            preferred_device='cpu',
        )
        h, r, t = numpy.asarray(triples_factory.mapped_triples).T
        if relation_margin:
            self.head_per_relation, self.tail_per_relation = [
                get_csr_matrix(
                    row_indices=r,
                    col_indices=col_indices,
                    shape=(triples_factory.num_relations, triples_factory.num_entities),
                )
                for col_indices in (h, t)
            ]
        else:
            self.head_per_relation = self.tail_per_relation = None
        if entity_margin:
            self.head_per_tail, self.tail_per_head = [
                get_csr_matrix(
                    row_indices=row_indices,
                    col_indices=col_indices,
                    shape=(triples_factory.num_entities, triples_factory.num_entities),
                )
                for row_indices, col_indices in ((t, h), (h, t))
            ]
        else:
            self.head_per_tail = self.tail_per_head = None

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        return _score(
            entity_relation_batch=hr_batch,
            per_entity=self.tail_per_head,
            per_relation=self.tail_per_relation,
            num_entities=self.num_entities,
        )

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        return _score(
            entity_relation_batch=rt_batch.flip(1),
            per_entity=self.head_per_tail,
            per_relation=self.head_per_relation,
            num_entities=self.num_entities,
        )


# TODO: Remove this model from this PR?
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


@click.command()
@verbose_option
@click.option('--batch-size', default=2048, show_default=True)
@click.option('--trials', default=10, show_default=True)
@click.option('--rebuild', is_flag=True)
@click.option('--test', is_flag=True, help='Run on the 5 smallest datasets and output to different path')
def main(batch_size: int, trials: int, rebuild: bool, test: bool):
    """Run the baseline showcase."""
    path = TEST_BENCHMARK_PATH if test else BENCHMARK_PATH
    if not path.is_file() or rebuild or test:
        with logging_redirect_tqdm():
            df = _build(batch_size=batch_size, trials=trials, path=path, test=test)
    else:
        df = pd.read_csv(path, sep='\t')

    _plot(df, test=test)


def _melt(df: pd.DataFrame) -> pd.DataFrame:
    keep = [col for col in df.columns if col not in METRICS]
    return pd.melt(
        df[[*keep, *METRICS]],
        id_vars=keep,
        value_vars=METRICS,
        var_name='metric',
    )


def _plot(df: pd.DataFrame, skip_small: bool = True, test: bool = False) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if skip_small and not test:
        df = df[~df.dataset.isin({'Nations', 'Countries', 'UMLS', 'Kinships'})]

    tsdf = _melt(df)

    # Plot relation between dataset and time, stratified by model
    # Interpretation: exponential relationship between # triples and time
    g = sns.catplot(
        data=tsdf,
        y='dataset',
        x='time',
        hue='model',
        kind='box',
        aspect=1.5,
    ).set(xscale='log', xlabel='Time (seconds)', ylabel='')
    g.fig.savefig(PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_timeplot.svg'))
    g.fig.savefig(PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_timeplot.png'), dpi=300)
    plt.close(g.fig)

    # Show AMRI plots. Surprisingly, some performance is really good.
    g = sns.catplot(
        data=tsdf[tsdf.metric == 'aamri'],
        y='dataset',
        x='value',
        hue='model',
        kind='violin',
        aspect=1.5,
    ).set(xlabel='Adjusted Mean Rank Index', ylabel='')
    g.fig.savefig(PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_aamri.svg'))
    g.fig.savefig(PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_aamri.png'), dpi=300)
    plt.close(g.fig)

    # Make a violinplot grid showing relation between # triples and result, stratified by model and metric.
    # Interpretation: no dataset size dependence
    g = sns.catplot(
        data=tsdf[~tsdf.metric.isin({'aamr', 'aamri'})],
        y='dataset',
        x='value',
        hue='model',
        col='metric',
        kind="violin",
        col_wrap=2,
        height=0.5 * tsdf['dataset'].nunique(),
        aspect=1.5,
    )
    g.set(ylabel='')
    g.fig.savefig(PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_scatterplot.svg'))
    g.fig.savefig(PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_scatterplot.png'), dpi=300)
    plt.close(g.fig)


def _build(batch_size: int, trials: int, path: Union[str, Path], test: bool = False) -> pd.DataFrame:
    datasets = sorted(dataset_resolver, key=Dataset.triples_sort_key)
    if test:
        datasets = datasets[:4]
    else:
        # FB15K and CoDEx Large are the first datasets where this gets a bit out of hand
        datasets = datasets[:1 + datasets.index(dataset_resolver.lookup('FB15k'))]
    models_kwargs: List[Tuple[Type[EvaluationOnlyModel], Mapping[str, Any]]] = [
        (MarginalDistributionBaseline, dict(entity_margin=True, relation_margin=True)),
        (MarginalDistributionBaseline, dict(entity_margin=True, relation_margin=False)),
        (MarginalDistributionBaseline, dict(entity_margin=False, relation_margin=True)),
        (SoftInverseTripleBaseline, dict(threshold=0.97)),
    ]
    kwargs_keys = sorted({k for _, d in models_kwargs for k in d})

    it = process_map(
        partial(
            _run_trials,
            batch_size=batch_size,
            kwargs_keys=kwargs_keys,
            trials=trials,
        ),
        itt.product(datasets, models_kwargs),
        desc='Baseline',
        total=len(datasets) * len(models_kwargs),
    )
    rows = list(itt.chain.from_iterable(it))
    columns = ['dataset', 'entities', 'relations', 'triples', 'trial', 'model', *kwargs_keys, 'time', *METRICS]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, sep='\t', index=False)
    print(tabulate(df.round(3).values, headers=columns, tablefmt='github'))
    return df


def _run_trials(
    t: Tuple[Type[Dataset], Tuple[Type[Model], Mapping[str, Any]]],
    *,
    trials: int,
    batch_size: int,
    kwargs_keys: Sequence[str],
) -> List[Tuple[Any, ...]]:
    dataset_cls, (model_cls, model_kwargs) = t

    model_name = model_cls.__name__[:-len('Baseline')]
    dataset_name = dataset_cls.__name__
    dataset = dataset_cls()
    base_record = (
        dataset_name,
        dataset.training.num_entities,
        dataset.training.num_relations,
        dataset.training.num_triples,
    )
    records = []
    for trial in trange(trials, leave=False, desc=f'{dataset_name}/{model_name}'):
        if trials != 0:
            trial_dataset = dataset.remix(random_state=trial)
        else:
            trial_dataset = dataset
        model = model_cls(triples_factory=trial_dataset.training, **model_kwargs)

        start_time = time.time()
        result = _evaluate_baseline(trial_dataset, model, batch_size=batch_size)
        elapsed_seconds = time.time() - start_time

        records.append((
            *base_record,
            trial,
            model_name,
            *(model_kwargs.get(key) for key in kwargs_keys),
            elapsed_seconds,
            *(result.get_metric(metric) for metric in METRICS),
        ))
    return records


def _evaluate_baseline(dataset: Dataset, model: Model, batch_size=None) -> RankBasedMetricResults:
    assert dataset.validation is not None
    evaluator = RankBasedEvaluator(ks=KS)
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
