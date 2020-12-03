# -*- coding: utf-8 -*-

"""Run the remixing experiments."""

import itertools as itt
import json
import logging
import os
import pickle
import random
from functools import partial
from typing import Type

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm.autonotebook import tqdm, trange

import pykeen.version
from pykeen.constants import PYKEEN_EXPERIMENTS
from pykeen.datasets import DataSet, get_dataset
from pykeen.models import Model, get_model_cls, models
from pykeen.pipeline import PipelineResult, pipeline
from pykeen.stoppers import EarlyStopper
from pykeen.triples.remix import dataset_splits_distance, remix_dataset, starmap_ctx
from pykeen.utils import normalize_string, resolve_device

__all__ = [
    'remixing',
]

REMIX_DIR = PYKEEN_EXPERIMENTS / 'remixing'
REMIX_DIR.mkdir(exist_ok=True, parents=True)


def _run(*, dataset, model, device, num_epochs, stopper=None) -> PipelineResult:
    return pipeline(
        model=model,
        dataset=dataset,
        random_seed=random.randint(1, 2 ** 32),
        device=device,
        stopper=stopper,
        stopper_kwargs=dict(patience=5, frequency=5),
        training_kwargs=dict(num_epochs=num_epochs, tqdm_kwargs=dict(leave=False), use_tqdm_batch=False),
        evaluation_kwargs=dict(tqdm_kwargs=dict(leave=False)),
    )


def _harness(
    dataset: str,
    model: str,
    *,
    device='cpu',
    trials: int = 30,
    overwrite: bool = False,
) -> str:
    metrics = ['adjusted_mean_rank', 'hits_at_10']

    logging.getLogger('pykeen.evaluation.evaluator').setLevel(logging.ERROR)
    logging.getLogger('pykeen.stoppers.early_stopping').setLevel(logging.ERROR)
    logging.getLogger('pykeen.triples.triples_factory').setLevel(logging.ERROR)
    logging.getLogger('pykeen.models.cli').setLevel(logging.ERROR)

    model: Type[Model] = get_model_cls(model)
    dataset: DataSet = get_dataset(dataset=dataset)
    dataset_norm_name = dataset.get_normalized_name()
    dataset_name = dataset.__class__.__name__
    model_name = model.__name__
    model_norm_name = normalize_string(model_name)
    device = resolve_device(device)

    output = os.path.join(REMIX_DIR, dataset_norm_name, model_norm_name)
    results_directory = os.path.join(output, 'results')
    os.makedirs(results_directory, exist_ok=True)

    reference_directory = os.path.join(output, 'reference.pkl')
    if not os.path.exists(reference_directory):
        tqdm.write(f'{dataset_name} / {model_name}')
        reference_results = _run(model=model, dataset=dataset, device=device, stopper='early', num_epochs=500)
        with open(reference_directory, 'wb') as file:
            pickle.dump(reference_results, file)
    else:
        with open(reference_directory, 'rb') as file:
            reference_results = pickle.load(file)

    assert isinstance(reference_results.stopper, EarlyStopper)
    reference_best_epochs = reference_results.stopper.best_epoch + 5  # add some wiggle room
    reference_metrics = {
        metric: reference_results.metric_results.get_metric(metric)
        for metric in metrics
    }

    rows = []
    for i in trange(1, 1 + trials, desc=f'{dataset_name} / {model_name}', unit='trial'):
        trial_results_path = os.path.join(results_directory, f'{i:04}.pkl')
        if os.path.exists(trial_results_path) and not overwrite:
            with open(trial_results_path, 'rb') as file:
                remixed_dataset, remixed_distance, remixed_results = pickle.load(file)
        else:
            remixed_dataset = remix_dataset(dataset, random_state=i)
            remixed_distance = dataset_splits_distance(dataset, remixed_dataset)
            remixed_results = _run(
                dataset=remixed_dataset,
                model=model,
                device=device,
                num_epochs=reference_best_epochs,
            )
            with open(trial_results_path, 'wb') as file:
                pickle.dump((remixed_dataset, remixed_distance, remixed_results), file)

        remixed_metrics = {
            metric: remixed_results.metric_results.get_metric(metric)
            for metric in reference_metrics
        }

        rows.append((
            dataset_norm_name,
            model_norm_name,
            i,
            remixed_distance,
            *itt.chain.from_iterable([
                (remixed_metrics[metric], reference_metrics[metric] - remixed_metrics[metric])
                for metric in metrics
            ]),
        ))

    df = pd.DataFrame(rows, columns=[
        'dataset',
        'model',
        'trial',
        'distance',
        *itt.chain.from_iterable([
            (metric, f'{metric}_diff')
            for metric in metrics
        ]),
    ])

    for metric in metrics:
        for x in ('distance', metric):
            fig, ax = plt.subplots(1, 1, figsize=(14, 6))
            sns.histplot(data=df, x=x, ax=ax)
            ax.set_title(f'Distribution of {dataset_name} Remix {x.capitalize()} to Reference')
            distribution_path = os.path.join(output, f'remix_{x}_distribution.png')
            tqdm.write(f'Outputting distribution to {distribution_path}')
            plt.savefig(distribution_path, dpi=300)
            plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.axhline(reference_metrics[metric])
        sns.scatterplot(data=df, x='distance', y=metric, ax=ax)
        ax.set_xlabel('Remix Distance')
        ax.set_ylabel(metric)
        ax.set_title(f'{dataset_name} / {model_name}')
        sp_path = os.path.join(output, f'remix_vs_{metric}.png')
        tqdm.write(f'Outputting scatterplot to {sp_path}')
        plt.savefig(sp_path, dpi=300)
        plt.close(fig)

    tests = {
        metric: stats.ttest_1samp(df[metric], reference_metrics[metric])
        for metric in metrics
    }
    extras_path = os.path.join(output, 'extras.json')
    tqdm.write(f'Outputting extras to {extras_path}')
    with open(extras_path, 'w') as file:
        json.dump(
            {
                'reference_best_epoch': reference_best_epochs,
                'reference_metrics': reference_metrics,
                'statistics': {k: v[0] for k, v in tests.items()},
                'p_values': {k: v[1] for k, v in tests.items()},
                'pykeen_hash': pykeen.version.get_git_hash(),
                'pykeen_version': pykeen.version.get_version(),
                'device': device.type,
            },
            file,
            indent=2,
        )

    rv = os.path.join(output, 'results.tsv')
    df.to_csv(rv, sep='\t', index=False)
    return rv


@click.command()
@click.option('--trials', type=int, default=100, show_default=True)
@click.option('--use-multiprocessing', is_flag=True)
def remixing(trials: int, use_multiprocessing: bool):
    """Run remixing experiments."""
    datasets = [
        'nations',
        # 'kinships',
        # 'codexsmall',
    ]
    model_blacklist = ['conve', 'convkb', 'tucker', 'rgcn', 'complexliteral', 'distmultliteral']
    # models = ['rotate', 'distmult', 'transe', 'complex', 'simple']

    pairs = list(itt.product(datasets, [model for model in sorted(models) if model not in model_blacklist]))
    partial_harness = partial(_harness, trials=trials)

    with starmap_ctx(use_multiprocessing) as ctx:
        paths = list(ctx.starmap(partial_harness, pairs))

    df = pd.concat([pd.read_csv(path, sep='\t') for path in paths])
    df.to_csv(os.path.join(REMIX_DIR, 'results.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    remixing()
