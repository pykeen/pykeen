# -*- coding: utf-8 -*-

"""Benchmark the speed for generating new datasets by remixing old ones."""

import itertools as itt
import logging
import os
import time
from datetime import datetime

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from humanize import intword
from tqdm import tqdm

from pykeen.datasets import dataset_resolver, get_dataset
from pykeen.triples.splitting import split
from pykeen.utils import get_benchmark
from pykeen.version import get_git_hash

SPLITTING_DIRECTORY = get_benchmark('splitting')
RESULTS_DIRECTORY = SPLITTING_DIRECTORY / 'results'
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

tsv_path = SPLITTING_DIRECTORY / 'split_benchmark.tsv'
png_path = SPLITTING_DIRECTORY / 'split_benchmark.png'
scatter_png_path = SPLITTING_DIRECTORY / 'split_benchmark_scatter.png'
columns = [
    'hash',
    'dataset',
    'dataset_size',
    'dataset_load_time',
    'dataset_cat_time',
    'method',
    'ratio',
    'replicate',
    'split_time',
    'training_size',
    'testing_size',
    'validation_size',
]


def _log(s):
    tqdm.write(f'[{datetime.now().strftime("%H:%M:%S")}] {s}')


@click.command()
@click.option('-r', '--replicates', type=int, default=5, show_default=True)
@click.option('-f', '--force', is_flag=True)
def main(replicates: int, force: bool):
    import pykeen.triples.splitting
    pykeen.triples.splitting.logger.setLevel(logging.ERROR)
    import pykeen.triples.triples_factory
    pykeen.triples.triples_factory.logger.setLevel(logging.ERROR)
    import pykeen.utils
    pykeen.utils.logger.setLevel(logging.ERROR)

    git_hash = get_git_hash()
    methods = ['cleanup', 'coverage']
    ratios = [0.8]

    click.echo(f'output directory: {SPLITTING_DIRECTORY.as_posix()}')
    rows = []
    outer_it = tqdm(sorted(dataset_resolver.lookup_dict), desc='Dataset')
    for dataset in outer_it:
        dataset_path = RESULTS_DIRECTORY / f'{dataset}.tsv'
        if dataset_path.exists() and not force:
            _log(f'loading pre-calculated {dataset} from {dataset_path}')
            df = pd.read_csv(dataset_path, sep='\t')
            rows.extend(df.values)
            continue

        _log(f'loading {dataset}')
        t = time.time()
        dataset = get_dataset(dataset=dataset)
        dataset_name = dataset.__class__.__name__
        ccl = [
            dataset.training.mapped_triples,
            dataset.testing.mapped_triples,
            dataset.validation.mapped_triples,
        ]
        load_time = time.time() - t
        _log(f'done loading {dataset_name} after {load_time:.3f} seconds')
        _log(f'concatenating {dataset_name}')
        t = time.time()
        mapped_triples: torch.LongTensor = torch.cat(ccl, dim=0)
        concat_time = time.time() - t
        _log(f'done concatenating {dataset_name} after {concat_time:.3f} seconds')
        _log(f'deleting {dataset_name}')
        del dataset
        _log(f'done deleting {dataset_name}')

        dataset_rows = []
        inner_it = itt.product(methods, ratios, range(1, 1 + replicates))
        inner_it = tqdm(
            inner_it,
            total=len(methods) * len(ratios) * replicates,
            desc=f'{dataset_name} ({intword(mapped_triples.shape[0])})',
        )
        for method, ratio, replicate in inner_it:
            t = time.time()
            results = split(
                mapped_triples,
                ratios=[ratio, (1 - ratio) / 2],
                method=method,
                random_state=replicate,
            )
            split_time = time.time() - t
            dataset_rows.append((
                git_hash,
                dataset_name,
                mapped_triples.shape[0],
                load_time,
                concat_time,
                method,
                ratio,
                replicate,
                split_time,
                results[0].shape[0],
                results[1].shape[0],
                results[2].shape[0],
            ))
            del results

        _log(f'writing to {dataset_path}')
        pd.DataFrame(dataset_rows, columns=columns).to_csv(dataset_path, sep='\t', index=False)
        rows.extend(dataset_rows)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(tsv_path, sep='\t', index=False)
    _make_1(df, git_hash)
    _make_2(df, git_hash)


def _make_1(df, git_hash):
    """Make the chart comparing the dataset times by method."""
    fig, ax = plt.subplots(1, 1)
    sns.barplot(data=df, y='dataset', x='split_time', hue='method', ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel('Split Time (s)')
    ax.set_ylabel('')
    ax.set_title(git_hash)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def _make_2(df, git_hash):
    """Make chart comparing dataset sizes to times."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey='all')
    xs = [
        'dataset_size',
        'training_size',
        'testing_size',
    ]
    for x, ax in zip(xs, axes.ravel()):
        sns.scatterplot(
            data=df, y='split_time', x=x, hue='dataset', style='method', ax=ax, x_jitter=.1,
            legend=None,
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Split Time (s)')
        ax.set_title(git_hash)

    fig.tight_layout()
    fig.savefig(scatter_png_path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
