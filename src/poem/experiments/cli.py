# -*- coding: utf-8 -*-

"""Run landmark experiments."""

import itertools as itt
import json
import logging
import os
import shutil
import sys
import time
from typing import Optional
from uuid import uuid4

import click

from ..utils import normalize_string

__all__ = [
    'experiment',
]

logger = logging.getLogger(__name__)
HERE = os.path.abspath(os.path.dirname(__file__))


def _turn_on_debugging(_ctx, _param, value):
    if value:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)


verbose_option = click.option(
    '-v', '--verbose',
    is_flag=True,
    expose_value=False,
    callback=_turn_on_debugging,
)
directory_option = click.option(
    '-d', '--directory',
    type=click.Path(dir_okay=True, exists=True, file_okay=False),
    default=os.getcwd(),
)


@click.group()
def experiment():
    """Run landmark experiments."""


@experiment.command()
@click.argument('model')
@click.argument('reference')
@click.argument('dataset')
@directory_option
def reproduce(model: str, reference: str, dataset: str, directory: str):
    """Reproduce a pre-defined experiment included in PyKEEN.

    Example: python -m poem.experiments reproduce tucker balazevic2019 fb15k
    """
    file_name = f'{reference}_{model}_{dataset}'
    path = os.path.join(HERE, model, f'{file_name}.json')
    _help_reproduce(directory=directory, path=path, file_name=file_name)


@experiment.command()
@click.argument('path')
@directory_option
def run(path: str, directory: str):
    """Run a single reproduction experiment."""
    _help_reproduce(directory=directory, path=path)


def _help_reproduce(*, directory, path, file_name=None) -> None:
    """Help run the configuration at a given path.

    :param directory: Output directory
    :param path: Path to configuration JSON file
    :param file_name: Name of JSON file (optional)
    """
    from poem.pipeline import pipeline_from_path

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


@experiment.command()
@click.argument('path')
@verbose_option
@click.option('-d', '--directory', type=click.Path(file_okay=False, dir_okay=True))
def optimize(path: str, directory: str):
    """Run a single HPO experiment."""
    _run_hpo_helper(path, output_directory=directory)


@experiment.command()
@click.argument('path', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@directory_option
@click.option('--dry-run', is_flag=True)
@verbose_option
def generate_hpo_studies(path: str, directory: Optional[str], dry_run: bool) -> None:
    """Generate a set of HPO configurations."""
    from poem.training import _TRAINING_LOOP_SUFFIX

    datetime = time.strftime('%Y-%m-%d-%H-%M')
    directory = os.path.join(directory, f'{datetime}_{uuid4()}')

    with open(path) as file:
        config = json.load(file)

    metadata = config['metadata']
    optuna_config = config['optuna']
    ablation_config = config['ablation']

    evaluator = ablation_config['evaluator']
    evaluator_kwargs = ablation_config['evaluator_kwargs']
    evaluation_kwargs = ablation_config['evaluation_kwargs']

    it = itt.product(
        ablation_config['datasets'],
        ablation_config['create_inverse_triples'],
        ablation_config['models'],
        ablation_config['loss_functions'],
        ablation_config['regularizers'],
        ablation_config['optimizers'],
        ablation_config['training_loops'],
    )

    for counter, (
        dataset,
        create_inverse_triples,
        model,
        loss,
        regularizer,
        optimizer,
        training_loop,
    ) in enumerate(it):
        experiment_name = f'{counter:03d}_{normalize_string(dataset)}_{normalize_string(model)}'
        output_directory = os.path.join(directory, experiment_name)
        os.makedirs(output_directory, exist_ok=True)
        # TODO what happens if already exists?

        hpo_config = dict(
            # TODO incorporate setting of random seed
            # pipeline_kwargs=dict(
            #    random_seed=random.randint(1, 2 ** 32 - 1),
            # ),
        )

        def _set_arguments(key: str, value: str) -> None:
            """Set argument and its values."""
            d = {key: value}
            kwargs = ablation_config[f'{key}_kwargs'][model][value]
            if kwargs:
                d[f'{key}_kwargs'] = kwargs
            kwargs_ranges = ablation_config[f'{key}_kwargs_ranges'][model][value]
            if kwargs_ranges:
                d[f'{key}_kwargs_ranges'] = kwargs_ranges

            hpo_config.update(d)

        # Add dataset to current_pipeline
        hpo_config['dataset'] = dataset
        logger.info(f"Dataset: {dataset}")
        hpo_config['dataset_kwargs'] = dict(create_inverse_triples=create_inverse_triples)
        logger.info(f"Add inverse triples: {create_inverse_triples}")

        hpo_config['model'] = model
        model_kwargs = ablation_config['model_kwargs'][model]
        if model_kwargs:
            hpo_config['model_kwargs'] = ablation_config['model_kwargs'][model]
        hpo_config['model_kwargs_ranges'] = ablation_config['model_kwargs_ranges'][model]
        logger.info(f"Model: {model}")

        # Add loss function to current_pipeline
        _set_arguments(key='loss', value=loss)
        logger.info(f"Loss function: {loss}")

        # Add regularizer to current_pipeline
        _set_arguments(key='regularizer', value=regularizer)
        logger.info(f"Regularizer: {regularizer}")

        # Add optimizer to current_pipeline
        _set_arguments(key='optimizer', value=optimizer)
        logger.info(f"Optimizer: {optimizer}")

        # Add training assumption to current_pipeline
        hpo_config['training_loop'] = training_loop
        logger.info(f"Training loop: {training_loop}")

        if normalize_string(training_loop, suffix=_TRAINING_LOOP_SUFFIX) == 'owa':
            negative_sampler = ablation_config['negative_sampler']
            _set_arguments(key='negative_sampler', value=negative_sampler)
            logger.info(f"Negative sampler: {negative_sampler}")

        # Add training kwargs and kwargs_ranges
        training_kwargs = ablation_config['training_kwargs'][model][training_loop]
        if training_kwargs:
            hpo_config['training_kwargs'] = training_kwargs
        hpo_config['training_kwargs_ranges'] = ablation_config['training_kwargs_ranges'][model][
            training_loop]

        # Add evaluation
        hpo_config['evaluator'] = evaluator
        if evaluator_kwargs:
            hpo_config['evaluator_kwargs'] = evaluator_kwargs
        hpo_config['evaluation_kwargs'] = evaluation_kwargs
        logger.info(f"Evaluator: {evaluator}")

        config_path = os.path.join(output_directory, 'hpo_config.json')
        with open(config_path, 'w') as file:
            json.dump(dict(
                metadata=metadata,
                pipeline=hpo_config,
                optuna=optuna_config,
            ), file, indent=2, ensure_ascii=True)

        if not dry_run:
            _run_hpo_helper(config_path, output_directory=output_directory)


def _run_hpo_helper(path: str, *, output_directory: Optional[str] = None):
    from poem.hpo import hpo_pipeline_from_path

    hpo_pipeline_result = hpo_pipeline_from_path(path)
    hpo_pipeline_result.dump_to_directory(output_directory)


if __name__ == '__main__':
    experiment()
