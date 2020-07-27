# -*- coding: utf-8 -*-

"""Utilities for ablation study configurations."""

import itertools as itt
import json
import logging
import os
import sys
import time
from copy import deepcopy
from typing import Any, Mapping, Optional
from uuid import uuid4

from ..training import _TRAINING_LOOP_SUFFIX
from ..utils import normalize_string

__all__ = [
    'ablation_pipeline',
    'prepare_ablation_from_config',
    'prepare_ablation',
]

logger = logging.getLogger(__name__)


def ablation_pipeline(
    config: Mapping[str, Any],
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
    datetime = time.strftime('%Y-%m-%d-%H-%M')
    directory = os.path.join(directory, f'{datetime}_{uuid4()}')

    directories = prepare_ablation_from_config(config=config, directory=directory, save_artifacts=save_artifacts)
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
        logger.info('Re-training best pipeline and saving artifacts in %s', best_pipeline_dir)
        hpo_pipeline_result.replicate_best_pipeline(
            replicates=best_replicates,
            move_to_cpu=move_to_cpu,
            save_replicates=not discard_replicates,
            directory=best_pipeline_dir,
        )


def prepare_ablation(path: str, directory: str, save_artifacts: bool):
    """Prepare a set of ablation study directories.

    TODO @mehdi documentation of parameters
    TODO @mehdi type annotation for return type
    """
    with open(path) as file:
        config = json.load(file)
    return prepare_ablation_from_config(config=config, directory=directory, save_artifacts=save_artifacts)


def prepare_ablation_from_config(config: Mapping[str, Any], directory: str, save_artifacts: bool):
    """Prepare a set of ablation study directories.

    TODO @mehdi documentation of parameters
    TODO @mehdi type annotation for return type
    """
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

    directories = []
    for counter, (
        dataset,
        create_inverse_triples,
        model,
        loss,
        regularizer,
        optimizer,
        training_loop,
    ) in enumerate(it):
        dataset_name = normalize_string(dataset) if isinstance(dataset, str) else 'user_data'
        experiment_name = f'{counter:04d}_{dataset_name}_{normalize_string(model)}'
        output_directory = os.path.join(directory, experiment_name)
        os.makedirs(output_directory, exist_ok=True)
        # TODO what happens if already exists?

        _experiment_optuna_config = optuna_config.copy()
        _experiment_optuna_config['storage'] = f'sqlite:///{output_directory}/optuna_results.db'
        if save_artifacts:
            save_model_directory = os.path.join(output_directory, 'artifacts')
            os.makedirs(save_model_directory, exist_ok=True)
            _experiment_optuna_config['save_model_directory'] = save_model_directory

        hpo_config = dict()
        for retain_key in ('stopper', 'stopper_kwargs'):
            if retain_key in ablation_config:
                logger.info(f'Retaining {retain_key} configuration in HPO')
                hpo_config[retain_key] = deepcopy(ablation_config[retain_key])

        for error_key in ('early_stopping', 'early_stopping_kwargs'):
            if error_key in ablation_config:
                raise ValueError(f'Outdated key: {error_key}. Please update')

        # TODO incorporate setting of random seed
        # pipeline_kwargs=dict(
        #    random_seed=random.randint(1, 2 ** 32 - 1),
        # ),

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
        if isinstance(dataset, str):
            hpo_config['dataset'] = dataset
        elif isinstance(dataset, dict):
            # Training, test, and validation paths are provided
            hpo_config['training_triples_factory'] = dataset['training_triples_factory']
            hpo_config['testing_triples_factory'] = dataset['testing_triples_factory']
            hpo_config['validation_triples_factory'] = dataset['validation_triples_factory']
        else:
            TypeError("Dataset must be either the dataset name, i.e., of type str, or a dictionary containing\n"
                      "the paths to the training, testing, and validation data.")
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

        # Add training approach to current_pipeline
        hpo_config['training_loop'] = training_loop
        logger.info(f"Training loop: {training_loop}")

        if normalize_string(training_loop, suffix=_TRAINING_LOOP_SUFFIX) == 'slcwa':
            negative_sampler = ablation_config['negative_sampler']
            _set_arguments(key='negative_sampler', value=negative_sampler)
            logger.info(f"Negative sampler: {negative_sampler}")

        # Add training kwargs and kwargs_ranges
        training_kwargs = ablation_config['training_kwargs'][model][training_loop]
        if training_kwargs:
            hpo_config['training_kwargs'] = training_kwargs
        hpo_config['training_kwargs_ranges'] = ablation_config['training_kwargs_ranges'][model][training_loop]

        # Add evaluation
        hpo_config['evaluator'] = evaluator
        if evaluator_kwargs:
            hpo_config['evaluator_kwargs'] = evaluator_kwargs
        hpo_config['evaluation_kwargs'] = evaluation_kwargs
        logger.info(f"Evaluator: {evaluator}")

        rv_config = dict(
            type='hpo',
            metadata=metadata,
            pipeline=hpo_config,
            optuna=_experiment_optuna_config,
        )

        rv_config_path = os.path.join(output_directory, 'hpo_config.json')
        with open(rv_config_path, 'w') as file:
            json.dump(rv_config, file, indent=2, ensure_ascii=True)

        directories.append((output_directory, rv_config_path))

    return directories
