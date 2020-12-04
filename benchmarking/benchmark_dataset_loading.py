# -*- coding: utf-8 -*-

"""Benchmark the speed for generating new datasets by remixing old ones."""

from datetime import datetime

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from tqdm import tqdm

from pykeen.datasets import datasets
from pykeen.utils import get_benchmark
from pykeen.version import get_git_hash

PYKEEN_BENCHMARK_DATASET_LOADING = get_benchmark('dataset_loading')
PATH = PYKEEN_BENCHMARK_DATASET_LOADING / 'times.tsv'
PNG_PATH = PYKEEN_BENCHMARK_DATASET_LOADING / 'times.png'


def _log(s):
    tqdm.write(f'[{datetime.now().strftime("%H:%M:%S")}] {s}')


@click.command()
@click.option('--replicates', type=int, default=2, show_default=True)
def main(replicates: int):
    """Benchmark loading times for datasets."""
    git_hash = get_git_hash()
    rows = []
    for replicate in range(1, 1 + replicates):
        for name, dataset_cls in tqdm(sorted(datasets.items()), desc=f'Replicate {replicate}'):
            _log(f'[r{replicate}] loading {name}')
            start_time = time.time()
            x = dataset_cls()
            _ = x.training, x.testing, x.validation
            end_time = time.time() - start_time
            rows.append((git_hash, x.__class__.__name__, replicate, end_time))
            del x

    df = pd.DataFrame(rows, columns=[
        'hash',
        'dataset',
        'replicate'
        'time',
    ])
    df.to_csv(PATH, sep='\t', index=False)

    fig, ax = plt.subplots(1, 1)
    sns.barplot(data=df, x='dataset', y='time', ax=ax)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
