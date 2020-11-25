# -*- coding: utf-8 -*-

"""Run the remixing experiments."""

import itertools as itt
import json
import logging
import multiprocessing as mp
import os
import random
from contextlib import nullcontext
from functools import partial
from typing import Type

import click
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import trange

from pykeen.constants import PYKEEN_HOME
from pykeen.datasets import DataSet, get_dataset
from pykeen.models import Model, get_model_cls
from pykeen.pipeline import pipeline
from pykeen.stoppers import EarlyStopper
from pykeen.triples.remix import dataset_splits_distance, remix_dataset
from pykeen.utils import normalize_string, resolve_device

REMIX_DIR = os.path.join(PYKEEN_HOME, 'experiments', 'remixing')
os.makedirs(REMIX_DIR, exist_ok=True)


def _run(*, dataset, model, device, num_epochs, stopper):
    return pipeline(
        model=model,
        dataset=dataset,
        random_seed=random.randint(1, 2 ** 32),
        device=device,
        stopper=stopper,
        stopper_kwargs=dict(patience=5, frequency=5),
        training_kwargs=dict(num_epochs=num_epochs, tqdm_kwargs=dict(leave=False)),
        evaluation_kwargs=dict(tqdm_kwargs=dict(leave=False)),
    )


def harness(
    dataset: str,
    model: str,
    *,
    device='cpu',
    trials: int = 30,
    metric: str = 'hits_at_k',
    overwrite: bool = False,
):
    logging.getLogger('pykeen.evaluation.evaluator').setLevel(logging.WARNING)
    logging.getLogger('pykeen.stoppers.early_stopping').setLevel(logging.WARNING)
    model: Type[Model] = get_model_cls(model)
    dataset: DataSet = get_dataset(dataset=dataset)
    device = resolve_device(device)

    output = os.path.join(REMIX_DIR, dataset.get_normalized_name(), normalize_string(model.__name__))
    os.makedirs(output, exist_ok=True)

    reference_results = _run(model=model, dataset=dataset, device=device, stopper='early', num_epochs=500)
    assert isinstance(reference_results.stopper, EarlyStopper)
    reference_best_epochs = reference_results.stopper.best_epoch + 5  # add some wiggle room
    reference_metric = reference_results.metric_results.get_metric(metric)

    remix_distances = []
    remix_metrics = []
    for random_state in trange(trials, desc=f'{dataset.__class__.__name__} / {model.__name__}', unit='trial'):
        trial_directory = os.path.join(output, 'results', f'{random_state:04}')
        if os.path.exists(trial_directory) and not overwrite:
            continue  # already done this trial
        remixed_dataset = remix_dataset(dataset, random_state=random_state)
        remixed_distance = dataset_splits_distance(dataset, remixed_dataset)
        remixed_results = _run(
            dataset=remixed_dataset,
            model=model,
            device=device,
            stopper=None,
            num_epochs=reference_best_epochs,
        )
        remixed_results.save_to_directory(trial_directory)
        remix_distances.append(remixed_distance)
        remix_metrics.append(remixed_results.metric_results.get_metric(metric))

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    sns.histplot(remix_distances, ax=ax)
    ax.set_title(f'Distribution of {dataset.__class__.__name__} Remix Distances to Reference')
    distribution_path = os.path.join(output, 'remix_distance_distribution.png')
    click.echo(f'Outputting distribution to {distribution_path}')
    plt.savefig(distribution_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.axhline(reference_metric)
    sns.scatterplot(x=remix_distances, y=remix_metrics, ax=ax)
    ax.set_xlabel('Remix Distance')
    ax.set_ylabel(metric)
    ax.set_title(f'{dataset.__class__.__name__} / {reference_results.model.__class__.__name__}')
    sp_path = os.path.join(output, f'remix_vs_{metric}.png')
    click.echo(f'Outputting scatterplot to {sp_path}')
    plt.savefig(sp_path, dpi=300)
    plt.close(fig)

    statistic, p_value = stats.ttest_1samp(remix_metrics, reference_metric)
    extras_path = os.path.join(output, 'extras.json')
    click.echo(f'Outputting extras to {extras_path}')
    with open(extras_path, 'w') as file:
        json.dump(
            {
                'statistic': statistic,
                'p_value': p_value,
            },
            file,
            indent=2,
        )


@click.command()
@click.option('--trials', type=int, default=30, show_default=True)
@click.option('--metric', default='hits_at_k', show_default=True)
def remix(trials: int, metric: str):
    datasets = [
        'nations',
        # 'kinships',
        # 'codexsmall',
    ]
    models = ['rotate', 'distmult', 'transe', 'complex', 'simple', 'tucker']

    pairs = list(itt.product(datasets, models))
    partial_harness = partial(harness, trials=trials, metric=metric)

    manager = nullcontext(itt) if True else mp.Pool(mp.cpu_count() - 1)
    with manager as ctx:
        list(ctx.starmap(partial_harness, pairs))


if __name__ == '__main__':
    remix()
