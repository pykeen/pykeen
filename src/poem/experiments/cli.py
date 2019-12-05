# -*- coding: utf-8 -*-

"""Run landmark experiments."""

import os
import shutil
import sys
import time

import click

from poem.pipeline import pipeline_from_path

__all__ = [
    'main',
]

HERE = os.path.abspath(os.path.dirname(__file__))


@click.group()
def main():
    """Run landmark experiments."""


_directory_option = click.option(
    '-d', '--directory',
    type=click.Path(dir_okay=True, exists=True, file_okay=False),
    default=os.getcwd(),
)


@main.command()
@click.argument('model')
@click.argument('reference')
@click.argument('dataset')
@_directory_option
def reproduce(model: str, reference: str, dataset: str, directory: str):
    """Reproduce a pre-defined experiment included in PyKEEN.

    Example: python -m poem.experiments reproduce tucker balazevic2019 fb15k
    """
    file_name = f'{reference}_{model}_{dataset}'
    path = os.path.join(HERE, model, f'{file_name}.json')
    _help_reproduce(directory=directory, path=path, file_name=file_name)


@main.command()
@click.argument('path')
@_directory_option
def run(path: str, directory: str):
    """Run a single experiment."""
    _help_reproduce(directory=directory, path=path)


def _help_reproduce(*, directory, path, file_name=None) -> None:
    """Help run the configuration at a given path.

    :param directory: Output directory
    :param path: Path to configuration JSON file
    :param file_name: Name of JSON file (optional)
    """
    if not os.path.exists(path):
        click.secho(f'Could not find configuration at {path}', fg='red')
        return sys.exit(1)
    click.echo(f'Running configuration at {path}')
    pipeline_result = pipeline_from_path(path)

    # Create directory in which all experimental artifacts are saved
    if file_name is not None:
        output_directory = os.path.join(directory, time.strftime(f"%Y-%m-%d-%H-%M-%S_{file_name}"))
    else:
        output_directory = os.path.join(directory, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(output_directory, exist_ok=True)

    pipeline_result.save_to_directory(output_directory)
    shutil.copyfile(path, os.path.join(output_directory, 'configuration_copied.json'))


if __name__ == '__main__':
    main()
