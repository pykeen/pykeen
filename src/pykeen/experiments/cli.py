# -*- coding: utf-8 -*-

"""Run landmark experiments."""

import logging
import os
import pathlib
import shutil
import sys
import time
from typing import Optional, Union
from uuid import uuid4

import click
from more_click import verbose_option

__all__ = [
    'experiments',
]

logger = logging.getLogger(__name__)
HERE = pathlib.Path(__file__).parent.resolve()


def _make_dir(_ctx, _param, value):
    os.makedirs(value, exist_ok=True)
    return value


directory_option = click.option(
    '-d', '--directory',
    type=click.Path(dir_okay=True, file_okay=False),
    callback=_make_dir,
    default=os.getcwd(),
)
replicates_option = click.option(
    '-r', '--replicates', type=int, default=1, show_default=True,
    help='Number of times to retrain the model.',
)
move_to_cpu_option = click.option('--move-to-cpu', is_flag=True, help='Move trained model(s) to CPU after training.')
discard_replicates_option = click.option(
    '--discard-replicates', is_flag=True,
    help='Discard trained models after training.',
)


@click.group()
def experiments():
    """Run landmark experiments."""


@experiments.command()
@click.argument('model')
@click.argument('reference')
@click.argument('dataset')
@replicates_option
@move_to_cpu_option
@discard_replicates_option
@directory_option
@verbose_option
def reproduce(
    model: str,
    reference: str,
    dataset: str,
    replicates: int,
    directory: str,
    move_to_cpu: bool,
    discard_replicates: bool,
):
    """Reproduce a pre-defined experiment included in PyKEEN.

    Example: $ pykeen experiments reproduce tucker balazevic2019 fb15k
    """
    file_name = f'{reference}_{model}_{dataset}'
    path = HERE.joinpath(model, file_name).with_suffix('.json')
    _help_reproduce(
        directory=directory,
        path=path,
        replicates=replicates,
        move_to_cpu=move_to_cpu,
        save_replicates=not discard_replicates,
        file_name=file_name,
    )


@experiments.command()
@click.argument('path')
@replicates_option
@move_to_cpu_option
@discard_replicates_option
@directory_option
def run(
    path: str,
    replicates: int,
    directory: str,
    move_to_cpu: bool,
    discard_replicates: bool,
):
    """Run a single reproduction experiment."""
    _help_reproduce(
        path=path,
        replicates=replicates,
        directory=directory,
        move_to_cpu=move_to_cpu,
        save_replicates=not discard_replicates,
    )


def _help_reproduce(
    *,
    directory: Union[str, pathlib.Path],
    path: Union[str, pathlib.Path],
    replicates: int,
    move_to_cpu: bool = False,
    save_replicates: bool = True,
    file_name: Optional[str] = None,
) -> None:
    """Help run the configuration at a given path.

    :param directory: Output directory
    :param path: Path to configuration JSON file
    :param replicates: How many times the experiment should be run
    :param move_to_cpu: Should the model be moved back to the CPU? Only relevant if training on GPU.
    :param save_replicates: Should the artifacts of the replicates be saved?
    :param file_name: Name of JSON file (optional)
    :return: None
    """
    from pykeen.pipeline import replicate_pipeline_from_path

    if isinstance(path, str):
        path = pathlib.Path(path).resolve()

    if not path.is_file():
        click.secho(f'Could not find configuration at {path}', fg='red')
        sys.exit(1)
    click.echo(f'Running configuration at {path}')

    # Create directory in which all experimental artifacts are saved
    datetime = time.strftime('%Y-%m-%d-%H-%M-%S')
    if file_name is not None:
        experiment_id = f'{datetime}_{uuid4()}_{file_name}'
    else:
        experiment_id = f'{datetime}_{uuid4()}'

    if isinstance(directory, str):
        directory = pathlib.Path(directory).resolve()
    output_directory = directory.joinpath(experiment_id)
    output_directory.mkdir(exist_ok=True, parents=True)

    replicate_pipeline_from_path(
        path=path,
        directory=output_directory,
        replicates=replicates,
        use_testing_data=True,
        move_to_cpu=move_to_cpu,
        save_replicates=save_replicates,
    )
    shutil.copyfile(path, os.path.join(output_directory, 'configuration_copied.json'))


@experiments.command()
@click.argument('path')
@verbose_option
@click.option('-d', '--directory', type=click.Path(file_okay=False, dir_okay=True))
def optimize(path: str, directory: str):
    """Run a single HPO experiment."""
    from pykeen.hpo import hpo_pipeline_from_path
    hpo_pipeline_result = hpo_pipeline_from_path(path)
    hpo_pipeline_result.save_to_directory(directory)


@experiments.command()
@click.argument('path', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@directory_option
@click.option('--dry-run', is_flag=True)
@click.option('-r', '--best-replicates', type=int, help='Number of times to retrain the best model.')
@move_to_cpu_option
@discard_replicates_option
@click.option('-s', '--save-artifacts', is_flag=True)
@verbose_option
def ablation(
    path: str,
    directory: str,
    dry_run: bool,
    best_replicates: int,
    save_artifacts: bool,
    move_to_cpu: bool,
    discard_replicates: bool,
) -> None:
    """Generate a set of HPO configurations.

    A sample file can be run with ``pykeen experiments ablation tests/resources/hpo_complex_nations.json``.
    """
    from ..ablation.ablation import prepare_ablation_from_path, _run_ablation_experiments
    directories = prepare_ablation_from_path(path=path, directory=directory, save_artifacts=save_artifacts)

    _run_ablation_experiments(
        directories=directories,
        best_replicates=best_replicates,
        dry_run=dry_run,
        move_to_cpu=move_to_cpu,
        discard_replicates=discard_replicates,
    )


if __name__ == '__main__':
    experiments()
