# -*- coding: utf-8 -*-

"""Run the deterioration experiments."""

import itertools as itt
import logging
import os
import pickle
from typing import Type

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.autonotebook import tqdm

from pykeen.constants import PYKEEN_HOME
from pykeen.datasets import DataSet, get_dataset
from pykeen.models import Model, get_model_cls
from pykeen.pipeline import pipeline
from pykeen.triples.remix import dataset_splits_distance, deteriorate_dataset
from pykeen.utils import normalize_string, resolve_device

__all__ = [
    'deteriorating',
]

DETERIORATION_DIR = os.path.join(PYKEEN_HOME, 'experiments', 'deteriorating')
os.makedirs(DETERIORATION_DIR, exist_ok=True)


def _helper(*, dataset: str, model: str, device):
    dataset: DataSet = get_dataset(dataset=dataset)
    dataset_name = dataset.__class__.__name__
    dataset_norm_name = dataset.get_normalized_name()
    model: Type[Model] = get_model_cls(model)
    model_name = model.__name__
    model_norm_name = normalize_string(model_name)
    device = resolve_device(device)

    overwrite = False
    replicates = 25
    num_epochs = 120
    dts = [100, 200, 300, 400, 500, 600, 700, 1000]

    output_directory = os.path.join(DETERIORATION_DIR, dataset_norm_name, model_norm_name)
    trials_directory = os.path.join(output_directory, 'results')
    os.makedirs(trials_directory, exist_ok=True)

    rows = []
    it = tqdm(itt.product(dts, range(1, 1 + replicates)), total=replicates * len(dts), desc='Deteriorating')
    for n, replicate in it:
        deteriorated_dataset = deteriorate_dataset(dataset, n=n)
        results_path = os.path.join(trials_directory, f'{n:04}_trial{replicate:04}.pkl')
        if os.path.exists(results_path) and not overwrite:
            with open(results_path, 'rb') as file:
                deteriorated_dataset, deteriorated_results = pickle.load(file)
        else:
            try:
                deteriorated_results = pipeline(
                    model=model,
                    dataset=deteriorated_dataset,
                    random_seed=replicate,
                    device=device,
                    stopper='early',
                    stopper_kwargs=dict(patience=5, frequency=5),
                    training_kwargs=dict(num_epochs=num_epochs, tqdm_kwargs=dict(leave=False), use_tqdm_batch=False),
                    evaluation_kwargs=dict(tqdm_kwargs=dict(leave=False)),
                )
                with open(results_path, 'wb') as file:
                    pickle.dump((deteriorated_dataset, deteriorated_results), file, protocol=pickle.HIGHEST_PROTOCOL)
            except IndexError:
                tqdm.write(f'Failure for {dataset_name} / {model_name} {n} {replicate}')
                continue

        deteriorated_distance = dataset_splits_distance(dataset, deteriorated_dataset)

        rows.append((
            dataset_norm_name,
            model_norm_name,
            n,
            100 * n / dataset.training.num_triples,
            replicate,
            deteriorated_distance,
            deteriorated_results.metric_results.get_metric('adjusted_mean_rank'),
            deteriorated_results.metric_results.get_metric('hits@10'),
        ))

    df = pd.DataFrame(rows, columns=[
        'dataset', 'model', 'deterioration', 'deterioration_percent',
        'trial', 'distance', 'adjusted_mean_rank', 'hits@10',
    ])
    df.to_csv(os.path.join(output_directory, 'results.tsv'))

    for metric in ['adjusted_mean_rank', 'hits@10', 'distance']:
        fig, ax = plt.subplots(1, 1)
        sns.scatterplot(data=df, x='deterioration', y=metric, ax=ax, x_jitter=True)
        plt.savefig(os.path.join(output_directory, f'deterioration_{metric}.png'), dpi=300)
        plt.close(fig)


@click.command()
def deteriorating():
    """Run the deterioration experiments."""
    logging.getLogger('pykeen.evaluation.evaluator').setLevel(logging.ERROR)
    logging.getLogger('pykeen.stoppers.early_stopping').setLevel(logging.ERROR)
    logging.getLogger('pykeen.triples.triples_factory').setLevel(logging.ERROR)
    logging.getLogger('pykeen.models.cli').setLevel(logging.ERROR)

    _helper(dataset='nations', model='rotate', device='cpu')


if __name__ == '__main__':
    deteriorating()
