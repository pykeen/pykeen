# -*- coding: utf-8 -*-

"""PyKEEN's command line interface."""

import json
import os

import click
from click_default_group import DefaultGroup

from pykeen.cli.prompt import prompt_config
from pykeen.predict import start_predictions_pipeline
from pykeen.run import run
from pykeen.utilities.summarize import summarize_results


@click.group(cls=DefaultGroup, default='train', default_if_no_args=True)
@click.version_option()
def main():
    """PyKEEN: A software for training and evaluating knowledge graph embeddings."""


@main.command()
@click.option('-c', '--config', type=click.File(), help='A PyKEEN JSON configuration file')
def train(config):
    """Train a KGE model."""
    if config is not None:
        config = json.load(config)
    else:
        config = prompt_config()

    run(config)


@main.command()
@click.option('-m', '--model-directory', type=click.Path(file_okay=False, dir_okay=True))
@click.option('-d', '--data-directory', type=click.Path(file_okay=False, dir_okay=True))
@click.option('-t', '--training-set-path', type=click.Path())
def predict(model_directory: str, data_directory: str, training_set_path: str):
    """Predict new links based on trained model."""

    remove_training_triples = False

    if training_set_path is not None:
        remove_training_triples = True,

    start_predictions_pipeline(model_directory, data_directory, remove_training_triples, training_set_path)


@main.command()
@click.option('-d', '--directory', type=click.Path(file_okay=False, dir_okay=True), default=os.getcwd())
@click.option('-o', '--output', type=click.File('w'))
def summarize(directory: str, output):
    """Summarize contents of training and evaluation."""
    summarize_results(directory, output)


@main.command()
def prefixes():
    """List registered prefixes."""
    from pykeen.constants import IMPORTERS
    for name, handler in sorted(IMPORTERS.items()):
        handler_doc = handler.__doc__.split('\n')[0]
        click.echo(f'{name}: {handler_doc}')


if __name__ == '__main__':
    main()
