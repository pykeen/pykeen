# -*- coding: utf-8 -*-

"""Run the remixing experiments."""

import itertools as itt
import json
import logging
import os
import random
from typing import Type

import click
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm

from pykeen.constants import PYKEEN_HOME
from pykeen.datasets import DataSet, get_dataset
from pykeen.models import Model, get_model_cls
from pykeen.pipeline import pipeline
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
    *,
    dataset: str,
    model: str,
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
    if os.path.exists(output) and not overwrite:
        return
    os.makedirs(output, exist_ok=True)

    random_states = [
        random.randint(1, 2 ** 32)
        for _ in range(trials)
    ]
    remixed_datasets = [
        remix_dataset(dataset, random_state=random_state)
        for random_state in random_states
    ]
    remix_distances = [
        dataset_splits_distance(dataset, remixed_dataset)
        for remixed_dataset in remixed_datasets
    ]

    reference_results = _run(model=model, dataset=dataset, device=device, stopper='early', num_epochs=500)
    reference_best_epochs = reference_results.stopper.best_epoch + 5  # add some wiggle room
    reference_metric = reference_results.metric_results.get_metric(metric)

    remix_results = [
        _run(dataset=remixed_dataset, model=model, device=device, stopper=None, num_epochs=reference_best_epochs)
        for remixed_dataset in tqdm(remixed_datasets, desc=f'Remixing {dataset.__class__.__name__}')
    ]

    for random_state, remix_result in zip(random_states, remix_results):
        remix_result.save_to_directory(os.path.join(output, 'results', str(random_state)))

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    sns.histplot(remix_distances, ax=ax)
    ax.set_title(f'Distribution of {dataset.__class__.__name__} Remix Distances to Reference')
    distribution_path = os.path.join(output, 'remix_distance_distribution.png')
    click.echo(f'Outputting distribution to {distribution_path}')
    plt.savefig(distribution_path, dpi=300)
    plt.close(fig)

    remix_metrics = [
        result.metric_results.get_metric(metric)
        for result in remix_results
    ]

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
    datasets = ['nations', 'kinship']
    models = ['rotate', 'distmult', 'transe']
    for dataset, model in itt.product(datasets, models):
        harness(model=model, dataset=dataset, trials=trials, metric=metric)


if __name__ == '__main__':
    remix()
