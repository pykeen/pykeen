import itertools as itt
import logging
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from pykeen.datasets import get_dataset
from pykeen.triples.splitting import split

HERE = os.path.abspath(os.path.dirname(__file__))


@click.command()
@click.option('-r', '--replicates', type=int, default=10, show_default=True)
def main(replicates: int):
    import pykeen.triples.splitting
    pykeen.triples.splitting.logger.setLevel(logging.ERROR)
    import pykeen.triples.triples_factory
    pykeen.triples.triples_factory.logger.setLevel(logging.ERROR)
    import pykeen.utils
    pykeen.utils.logger.setLevel(logging.ERROR)

    methods = ['new', 'old']
    ratios = [0.8]
    datasets = [
        'nations',
        'kinships',
        'umls',
        'codexsmall',
        'codexmedium',
        'codexlarge',
        'wn18rr',
    ]

    rows = []
    outer_it = tqdm(datasets, desc='Dataset')
    for _dataset in outer_it:
        dataset = get_dataset(dataset=_dataset)
        triples = np.concatenate([
            dataset.training.triples,
            dataset.testing.triples,
            dataset.validation.triples,
        ])
        inner_it = itt.product(methods, ratios, range(replicates))
        inner_it = tqdm(
            inner_it,
            total=len(methods) * len(ratios) * replicates,
            desc=f'{_dataset} ({triples.shape[0]})',
        )
        for method, ratio, replicate in inner_it:
            t = time.time()
            split(
                triples=triples,
                ratios=[ratio, (1 - ratio) / 2],
                method=method,
                random_state=replicate,
            )
            total = time.time() - t
            rows.append((dataset.__class__.__name__, method, ratio, replicate, total))

    df = pd.DataFrame(rows, columns=[
        'dataset',
        'method',
        'ratio',
        'replicate',
        'time',
    ])
    tsv_path = os.path.join(HERE, 'split_benchmark.tsv')
    df.to_csv(tsv_path, sep='\t', index=False)

    sns.barplot(data=df, y='dataset', x='time', hue='method')
    png_path = os.path.join(HERE, 'split_benchmark.png')
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)


if __name__ == '__main__':
    main()
