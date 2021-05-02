# -*- coding: utf-8 -*-

"""Run dataset CLI."""

import logging
import pathlib
from textwrap import dedent
from typing import Union

import click
from more_click import verbose_option
from tqdm import tqdm


@click.group()
def main():
    """Run the dataset CLI."""


@main.command()
@verbose_option
def summarize():
    """Load all datasets."""
    for name, dataset in _iter_datasets():
        click.secho(f'Loading {name}', fg='green', bold=True)
        try:
            dataset().summarize(show_examples=None)
        except Exception as e:
            click.secho(f'Failed {name}', fg='red', bold=True)
            click.secho(str(e), fg='red', bold=True)


def _iter_datasets():
    from . import datasets
    import docdata
    it = tqdm(
        sorted(
            datasets.items(),
            key=lambda pair: docdata.get_docdata(pair[1])['statistics']['triples'],
        ),
        desc='Datasets',
    )
    for k, v in it:
        it.set_postfix(name=k)
        yield k, v


@main.command()
@verbose_option
@click.option('--dataset')
@click.option('-f', '--force', is_flag=True)
@click.option('-a', '--all-datasets', is_flag=True)
@click.option('--countplots', is_flag=True)
@click.option('-d', '--directory', type=click.Path(dir_okay=True, file_okay=False, resolve_path=True))
def analyze(dataset, force: bool, all_datasets: bool, countplots: bool, directory):
    """Generate analysis."""
    if all_datasets:
        for name, dataset in _iter_datasets():
            _analyze(dataset, force, countplots, directory=directory)
    elif dataset:
        _analyze(dataset, force, countplots, directory=directory)
    else:
        raise ValueError


def _analyze(dataset, force, countplots, directory: Union[None, str, pathlib.Path]):
    from pykeen.datasets import get_dataset
    from pykeen.constants import PYKEEN_DATASETS
    from . import analysis
    from tqdm import tqdm
    import pandas as pd
    import docdata
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(dedent("""\
            Please install plotting dependencies by

                pip install pykeen[plotting]

            or directly by

                pip install matplotlib seaborn
        """)) from None

    # Raise matplotlib level
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    if directory is None:
        directory = PYKEEN_DATASETS
    else:
        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

    dataset_instance = get_dataset(dataset=dataset)
    d = directory.joinpath(dataset_instance.__class__.__name__.lower(), 'analysis')
    d.mkdir(parents=True, exist_ok=True)

    dfs = {}
    it = tqdm(analysis.__dict__.items(), leave=False, desc='Stats')
    for name, func in it:
        if not name.startswith('get') or not name.endswith('df'):
            continue
        it.set_postfix(func=name)
        key = name[len('get_'):-len('_df')]
        path = d.joinpath(key).with_suffix('.tsv')
        if path.exists() and not force:
            df = pd.read_csv(path, sep='\t')
        else:
            df = func(dataset=dataset_instance)
            df.to_csv(d.joinpath(key).with_suffix('.tsv'), sep='\t', index=False)
        dfs[key] = df

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        data=dfs['relation_injectivity'],
        x='head',
        y='tail',
        size='support',
        hue='support',
        ax=ax,
    )
    ax.set_title(f'{docdata.get_docdata(dataset_instance.__class__)["name"]} Relation Injectivity')
    fig.tight_layout()
    fig.savefig(d.joinpath('relation_injectivity.svg'))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        data=dfs['relation_functionality'],
        x='functionality',
        y='inverse_functionality',
        ax=ax,
    )
    ax.set_title(f'{docdata.get_docdata(dataset_instance.__class__)["name"]} Relation Functionality')
    fig.tight_layout()
    fig.savefig(d.joinpath('relation_functionality.svg'))
    plt.close(fig)

    if countplots:
        entity_count_df = (
            dfs['entity_count']
            .groupby('entity_label')
            .sum()
            .reset_index()
            .sort_values('count', ascending=False)
        )
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=entity_count_df, y='entity_label', x='count', ax=ax)
        ax.set_ylabel('')
        ax.set_xscale('log')
        fig.tight_layout()
        fig.savefig(d.joinpath('entity_counts.svg'))
        plt.close(fig)

        relation_count_df = (
            dfs['relation_count']
            .groupby('relation_label')
            .sum()
            .reset_index()
            .sort_values('count', ascending=False)
        )
        fig, ax = plt.subplots(1, 1)
        sns.barplot(data=relation_count_df, y='relation_label', x='count', ax=ax)
        ax.set_ylabel('')
        ax.set_xscale('log')
        fig.tight_layout()
        fig.savefig(d.joinpath('relation_counts.svg'))
        plt.close(fig)


if __name__ == '__main__':
    main()
