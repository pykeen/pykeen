# -*- coding: utf-8 -*-

"""Run the non-parametric baseline experiment.

Run with ``python -m pykeen.models.baseline.experiment``.
"""

import itertools as itt
import time
from functools import partial
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple, Type, Union, cast

import click
import pandas as pd
from more_click import verbose_option
from tabulate import tabulate
from tqdm import trange
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.logging import logging_redirect_tqdm

from pykeen.constants import PYKEEN_EXPERIMENTS
from pykeen.datasets import Dataset, dataset_resolver
from pykeen.models import Model
from pykeen.models.baseline.models import EvaluationOnlyModel, MarginalDistributionBaseline

BENCHMARK_PATH = PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark.tsv')
TEST_BENCHMARK_PATH = PYKEEN_EXPERIMENTS.joinpath('baseline_benchmark_test.tsv')
KS = (1, 5, 10, 50, 100)
METRICS = ['mrr', 'iamr', 'igmr', *(f'hits@{k}' for k in KS), 'aamr', 'aamri']


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
        # (SoftInverseTripleBaseline, dict(threshold=0.97)),
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


def _evaluate_baseline(dataset: Dataset, model: Model, batch_size=None):
    from pykeen.evaluation import RankBasedEvaluator, RankBasedMetricResults, evaluate
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
