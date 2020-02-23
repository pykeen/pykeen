# -*- coding: utf-8 -*-

"""Run landmark experiments."""

import logging
import os
import shutil
import sys
import time
from typing import Optional
from uuid import uuid4

import click

__all__ = [
    'experiments',
]

logger = logging.getLogger(__name__)
HERE = os.path.abspath(os.path.dirname(__file__))


def _turn_on_debugging(_ctx, _param, value):
    if value:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)


def _make_dir(_ctx, _param, value):
    os.makedirs(value, exist_ok=True)
    return value


verbose_option = click.option(
    '-v', '--verbose',
    is_flag=True,
    expose_value=False,
    callback=_turn_on_debugging,
)
directory_option = click.option(
    '-d', '--directory',
    type=click.Path(dir_okay=True, file_okay=False),
    callback=_make_dir,
    default=os.getcwd(),
)


@click.group()
def experiments():
    """Run landmark experiments."""


@experiments.command()
@click.argument('model')
@click.argument('reference')
@click.argument('dataset')
@click.option('--replicates', type=int, help='Number of times to retrain the model.')
@directory_option
def reproduce(model: str, reference: str, dataset: str, replicates: Optional[int], directory: str):
    """Reproduce a pre-defined experiment included in PyKEEN.

    Example: python -m pykeen.experiments reproduce tucker balazevic2019 fb15k
    """
    file_name = f'{reference}_{model}_{dataset}'
    path = os.path.join(HERE, model, f'{file_name}.json')
    _help_reproduce(directory=directory, path=path, replicates=replicates, file_name=file_name)


@experiments.command()
@click.argument('path')
@directory_option
def run(path: str, directory: str):
    """Run a single reproduction experiment."""
    _help_reproduce(directory=directory, path=path)


def _help_reproduce(*, directory: str, path: str, replicates: Optional[int] = None, file_name=None) -> None:
    """Help run the configuration at a given path.

    :param directory: Output directory
    :param path: Path to configuration JSON file
    :param file_name: Name of JSON file (optional)
    """
    from pykeen.pipeline import PipelineResultSet

    if not os.path.exists(path):
        click.secho(f'Could not find configuration at {path}', fg='red')
        return sys.exit(1)
    click.echo(f'Running configuration at {path}')

    pipeline_result_set = PipelineResultSet.from_path(path=path, replicates=replicates, use_testing_data=True)

    # Create directory in which all experimental artifacts are saved
    if file_name is not None:
        output_directory = os.path.join(directory, time.strftime(f"%Y-%m-%d-%H-%M-%S_{file_name}"))
    else:
        output_directory = os.path.join(directory, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(output_directory, exist_ok=True)

    pipeline_result_set.save_to_directory(output_directory)

    shutil.copyfile(path, os.path.join(output_directory, 'configuration_copied.json'))


@experiments.command()
@click.argument('path')
@verbose_option
@click.option('-d', '--directory', type=click.Path(file_okay=False, dir_okay=True))
def optimize(path: str, directory: str):
    """Run a single HPO experiment."""
    from pykeen.hpo import hpo_pipeline_from_path
    hpo_pipeline_result = hpo_pipeline_from_path(path)
    hpo_pipeline_result.dump_to_directory(directory)


@experiments.command()
@click.argument('path', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@directory_option
@click.option('--dry-run', is_flag=True)
@click.option('--no-retrain-best', is_flag=True)
@click.option('--best-replicates', type=int, help='Number of times to retrain the best model.')
@click.option('--save-artifacts', is_flag=True)
@verbose_option
def ablation(
    path: str,
    directory: Optional[str],
    dry_run: bool,
    no_retrain_best: bool,
    best_replicates: int,
    save_artifacts: bool,
) -> None:
    """Generate a set of HPO configurations.

    A sample file can be run with ``pykeen experiment ablation tests/resources/hpo_complex_nations.json``.
    """
    from pykeen.ablation import prepare_ablation

    datetime = time.strftime('%Y-%m-%d-%H-%M')
    directory = os.path.join(directory, f'{datetime}_{uuid4()}')

    directories = prepare_ablation(path=path, directory=directory, save_artifacts=save_artifacts)
    if dry_run:
        return sys.exit(0)

    from pykeen.hpo import hpo_pipeline_from_path

    for output_directory, rv_config_path in directories:
        hpo_pipeline_result = hpo_pipeline_from_path(rv_config_path)
        hpo_pipeline_result.dump_to_directory(output_directory)

        if no_retrain_best:
            continue

        best_pipeline_dir = os.path.join(output_directory, 'best_pipeline')
        os.makedirs(best_pipeline_dir, exist_ok=True)
        click.echo(f'Re-training best pipeline and saving artifacts in {best_pipeline_dir}')
        pipeline_result_set = hpo_pipeline_result.test_best_pipeline(replicates=best_replicates)
        pipeline_result_set.save_to_directory(best_pipeline_dir)


if __name__ == '__main__':
    experiments()
