# -*- coding: utf-8 -*-

"""Benchmark the speed for generating new datasets by remixing old ones."""

import itertools as itt
import logging
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from humanize import intword
from tqdm import tqdm

from pykeen.datasets import get_dataset
from pykeen.triples.splitting import split

HERE = os.path.abspath(os.path.dirname(__file__))
RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)

tsv_path = os.path.join(RESULTS, 'split_benchmark.tsv')
png_path = os.path.join(RESULTS, 'split_benchmark.png')
scatter_png_path = os.path.join(RESULTS, 'split_benchmark_scatter.png')


@click.command()
@click.option('-r', '--replicates', type=int, default=3, show_default=True)
def main(replicates: int):
    import pykeen.triples.splitting
    pykeen.triples.splitting.logger.setLevel(logging.ERROR)
    import pykeen.triples.triples_factory
    pykeen.triples.triples_factory.logger.setLevel(logging.ERROR)
    import pykeen.utils
    pykeen.utils.logger.setLevel(logging.ERROR)

    methods = ['old', 'new']
    ratios = [0.8]
    datasets = [
        'nations',
        'kinships',
        'umls',
        'codexsmall',
        'codexmedium',
        'codexlarge',
        'wn18rr',
        'FB15k237',
        'wn18',
        'fb15k',
        'YAGO310',
        'OGBBioKG',
        'hetionet',
        'OGBWikiKG',
        'openbiolink',
        'drkg',
    ]

    rows = []
    outer_it = tqdm(datasets, desc='Dataset')
    for dataset in outer_it:
        dataset = get_dataset(dataset=dataset)
        dataset_name = dataset.__class__.__name__
        triples = np.concatenate([
            dataset.training.triples,
            dataset.testing.triples,
            dataset.validation.triples,
        ])
        del dataset
        inner_it = itt.product(methods, ratios, range(1, 1 + replicates))
        inner_it = tqdm(
            inner_it,
            total=len(methods) * len(ratios) * replicates,
            desc=f'{dataset_name} ({intword(triples.shape[0])})',
        )
        for method, ratio, replicate in inner_it:
            t = time.time()
            results = split(
                triples=triples,
                ratios=[ratio, (1 - ratio) / 2],
                method=method,
                random_state=replicate,
            )
            total = time.time() - t
            rows.append((
                dataset_name,
                triples.shape[0],
                method,
                ratio,
                replicate,
                total,
                results[0].shape[0],
                results[1].shape[0],
                results[2].shape[0],
            ))
            del results

    df = pd.DataFrame(rows, columns=[
        'dataset',
        'dataset_size',
        'method',
        'ratio',
        'replicate',
        'time',
        'training_size',
        'testing_size',
        'validation_size',
    ])

    df.to_csv(tsv_path, sep='\t', index=False)
    _make_1(df)
    _make_2(df)


def _make_1(df):
    """Make the chart comparing the dataset times by method."""
    fig, ax = plt.subplots(1, 1)
    sns.barplot(data=df, y='dataset', x='time', hue='method', ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('')
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def _make_2(df):
    """Make chart comparing dataset sizes to times."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey='all')
    xs = [
        'dataset_size',
        'training_size',
        'testing_size',
    ]
    for x, ax in zip(xs, axes.ravel()):
        sns.scatterplot(
            data=df, y='time', x=x, hue='dataset', style='method', ax=ax, x_jitter=.1,
            legend=None,
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Time (s)')

    fig.tight_layout()
    fig.savefig(scatter_png_path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
