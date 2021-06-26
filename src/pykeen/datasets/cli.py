# -*- coding: utf-8 -*-

"""Run dataset CLI."""

import itertools as itt
import logging
import pathlib
from textwrap import dedent
from typing import Union

import click
import docdata
import pandas as pd
from more_click import verbose_option
from tqdm import tqdm

from . import dataset_resolver, get_dataset
from ..constants import PYKEEN_DATASETS


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


def _iter_datasets(regex_name_filter=None):
    it = sorted(
        dataset_resolver.lookup_dict.items(),
        key=lambda pair: docdata.get_docdata(pair[1])['statistics']['triples'],
    )
    if regex_name_filter is not None:
        if isinstance(regex_name_filter, str):
            import re
            regex_name_filter = re.compile(regex_name_filter)
        it = [
            (name, dataset)
            for name, dataset in it
            if regex_name_filter.match(name)
        ]
    it = tqdm(
        it,
        desc='Datasets',
    )
    for k, v in it:
        it.set_postfix(name=k)
        yield k, v


@main.command()
@verbose_option
@click.option('--dataset', help='Regex for filtering datasets by name')
@click.option('-f', '--force', is_flag=True)
@click.option('--countplots', is_flag=True)
@click.option('-d', '--directory', type=click.Path(dir_okay=True, file_okay=False, resolve_path=True))
def analyze(dataset, force: bool, countplots: bool, directory):
    """Generate analysis."""
    for _name, dataset in _iter_datasets(regex_name_filter=dataset):
        _analyze(dataset, force, countplots, directory=directory)


def _analyze(dataset, force, countplots, directory: Union[None, str, pathlib.Path]):
    from . import analysis

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(dedent("""\
            Please install plotting dependencies by

                pip install pykeen[plotting]

            or directly by

                pip install matplotlib seaborn
        """))

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


@main.command()
@verbose_option
@click.option('--dataset', help='Regex for filtering datasets by name')
def verify(dataset: str):
    """Verify dataset integrity."""
    data = []
    keys = None
    for name, dataset in _iter_datasets(regex_name_filter=dataset):
        dataset_instance = get_dataset(dataset=dataset)
        data.append(list(itt.chain(
            [name],
            itt.chain.from_iterable(
                (triples_factory.num_entities, triples_factory.num_relations)
                for _, triples_factory in sorted(dataset_instance.factory_dict.items())
            ),
        )))
        keys = keys or sorted(dataset_instance.factory_dict.keys())
    if not keys:
        return
    df = pd.DataFrame(
        data=data,
        columns=["name"] + [f"num_{part}_{a}" for part in keys for a in ("entities", "relations")],
    )
    valid = None
    for part, a in itt.product(("validation", "testing"), ("entities", "relations")):
        this_valid = df[f"num_training_{a}"] == df[f"num_{part}_{a}"]
        if valid is None:
            valid = this_valid
        else:
            valid = valid & this_valid
    df["valid"] = valid
    print(df.to_markdown())


if __name__ == '__main__':
    main()
