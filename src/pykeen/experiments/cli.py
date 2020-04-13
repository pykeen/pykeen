# -*- coding: utf-8 -*-

"""Run landmark experiments."""

import json
import logging
import os
import shutil
import sys
import time
from typing import Optional
from uuid import uuid4

import click

from ..datasets import get_dataset
from ..pipeline import pipeline_from_config, replicate_pipeline_from_config

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
    path = os.path.join(HERE, model, f'{file_name}.json')
    _help_reproduce(
        directory=directory,
        path=path,
        replicates=replicates,
        move_to_cpu=move_to_cpu,
        save_replicates=not discard_replicates,
        file_name=file_name,
    )


@experiments.command()
@click.option('-fb15k237', is_flag=True, help='Reproduce results for ConvKB on FB15K237.')
@click.option('-wn18rr', is_flag=True, help='Reproduce results for ConvKB on FB15K237.')
@replicates_option
@move_to_cpu_option
@discard_replicates_option
@directory_option
def reproduce_convkb(
    fb15k237: str,
    wn18rr: str,
    replicates: int,
    directory: str,
    move_to_cpu: bool,
    discard_replicates: str,
):
    """Tran ConvKB."""
    if fb15k237 and wn18rr:
        raise Exception('Cannot define both FB15k237 and WN18RR at the same time.')
    elif fb15k237:
        config_transe = os.path.join(HERE, 'convkb', 'nguyen2018_transe_fb15k237.json')
        config_convkb = os.path.join(HERE, 'convkb', 'nguyen2018_convkb_fb15k237.json')
    elif wn18rr:
        config_transe = os.path.join(HERE, 'convkb', 'nguyen2018_transe_wn18rr.json')
        config_convkb = os.path.join(HERE, 'convkb', 'nguyen2018_convkb_wn18rr.json')
    else:
        raise Exception('Either FB15K-237 (using \'-fb15k237\')or WN18RR (using \'-wn18rr\') has to be defined.')

    with open(config_transe) as file:
        config_transe = json.load(file)

    # Load TransE config
    with open(config_convkb) as file:
        config_convkb = json.load(file)

    # Train ConvKB
    training_triples_factory, testing_triples_factory, validation_triples_factory = get_dataset(
        dataset=config_convkb.get('pipeline').get('dataset'),
        dataset_kwargs=config_convkb.get('dataset_kwargs'),
    )

    del config_transe['pipeline']['dataset']

    pipeline_results = pipeline_from_config(
        config=config_transe,
        training_triples_factory=training_triples_factory,
        testing_triples_factory=testing_triples_factory,
    )
    trained_transe = pipeline_results.model

    model_kwargs = config_convkb.get('pipeline').get('model_kwargs')
    model_kwargs['entity_embeddings'] = trained_transe.entity_embeddings
    model_kwargs['relation_embeddings'] = trained_transe.relation_emebddings

    del config_convkb['pipeline']['dataset']

    replicate_pipeline_from_config(
        path=config_convkb,
        training_triples_factory=training_triples_factory,
        testing_triples_factory=testing_triples_factory,
        directory=directory,
        replicates=replicates,
        move_to_cpu=move_to_cpu,
        save_replicates=not discard_replicates,

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
    directory: str,
    path: str,
    replicates: int,
    move_to_cpu: bool = False,
    save_replicates: bool = True,
    file_name: Optional[str] = None,
) -> None:
    """Help run the configuration at a given path.

    :param directory: Output directory
    :param path: Path to configuration JSON file
    :param replicates: How many times the experiment should be run
    :param file_name: Name of JSON file (optional)
    """
    from pykeen.pipeline import replicate_pipeline_from_path

    if not os.path.exists(path):
        click.secho(f'Could not find configuration at {path}', fg='red')
        return sys.exit(1)
    click.echo(f'Running configuration at {path}')

    # Create directory in which all experimental artifacts are saved
    if file_name is not None:
        output_directory = os.path.join(directory, time.strftime(f"%Y-%m-%d-%H-%M-%S_{file_name}"))
    else:
        output_directory = os.path.join(directory, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(output_directory, exist_ok=True)

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
    directory: Optional[str],
    dry_run: bool,
    best_replicates: int,
    save_artifacts: bool,
    move_to_cpu: bool,
    discard_replicates: bool,
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
        hpo_pipeline_result.save_to_directory(output_directory)

        if not best_replicates:
            continue

        best_pipeline_dir = os.path.join(output_directory, 'best_pipeline')
        os.makedirs(best_pipeline_dir, exist_ok=True)
        click.echo(f'Re-training best pipeline and saving artifacts in {best_pipeline_dir}')
        hpo_pipeline_result.replicate_best_pipeline(
            replicates=best_replicates,
            move_to_cpu=move_to_cpu,
            save_replicates=not discard_replicates,
            directory=best_pipeline_dir,
        )


if __name__ == '__main__':
    experiments()
