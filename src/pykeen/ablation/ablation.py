# -*- coding: utf-8 -*-

"""Utilities for ablation study configurations."""

import itertools as itt
import json
import logging
import os
import time
from copy import deepcopy
from typing import Any, List, Mapping, Optional, Tuple, Union
from uuid import uuid4

from ..training import _TRAINING_LOOP_SUFFIX
from ..utils import normalize_string

__all__ = [
    'ablation_pipeline',
    'ablation_pipeline_from_config',
    'prepare_ablation_from_config',
    'prepare_ablation_from_path',
    'prepare_ablation',
]

logger = logging.getLogger(__name__)

Mapping2D = Mapping[str, Mapping[str, Any]]
Mapping3D = Mapping[str, Mapping[str, Mapping[str, Any]]]


def ablation_pipeline(
    datasets: Union[str, List[str]],
    models: Union[str, List[str]],
    losses: Union[str, List[str]],
    optimizers: Union[str, List[str]],
    training_loops: Union[str, List[str]],
    *,
    create_inverse_triples: Union[bool, List[bool]] = False,
    regularizers: Union[None, str, List[str]] = None,
    model_to_model_kwargs: Optional[Mapping2D] = None,
    model_to_model_kwargs_ranges: Optional[Mapping2D] = None,
    model_to_trainer_to_training_kwargs: Optional[Mapping3D] = None,
    model_to_trainer_to_training_kwargs_ranges: Optional[Mapping3D] = None,
    ablation_config: Optional[Mapping3D] = None,
    evaluator: Optional[str] = None,
    optuna_config: Optional[Mapping[str, Any]] = None,
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    directory: Optional[str] = None,
    dry_run: bool = False,
    best_replicates: Optional[int] = None,
    save_artifacts: bool = True,
    move_to_cpu: bool = True,
    discard_replicates: bool = False,
) -> None:
    """Generate a set of HPO configurations.

    A sample file can be run with``pykeen experiment ablation tests/resources/hpo_complex_nations.json``.

    :param datasets: A dataset name or list of dataset names
    :param models: A model name or list of model names
    :param losses: A loss function name or list of loss function names
    :param optimizers: An optimizer name or list of optimizer names
    :param training_loops: A training loop name or list of training loop names
    :param create_inverse_triples: Either a boolean for a single entry or a list of booleans
    :param regularizers: A regularizer name, list of regularizer names, or None if no regularizer is desired.
        Defaults to None.

    :param evaluator: The name of the evaluator to be used. Defaults to rank-based evaluator.
    :param evaluator_kwargs: The keyword arguments passed to the evaluator (in the pipeline)
    :param evaluation_kwargs: The keyword arguments passed during evaluation (in the pipeline)

    :param model_to_model_kwargs: A mapping from model name to dictionaries of default keyword arguments for
        the instantiation of that model
    :param model_to_model_kwargs_ranges: A mapping from model name to dictionaries of keyword argument
        ranges for that model to be used in HPO.
    :param model_to_trainer_to_training_kwargs: A mapping from model name to a mapping of trainer name to a mapping
        of default keyword arguments for the instantiation of that trainer. This is useful becuase for some models,
        you might want to set the number of epochs differently.
    :param model_to_trainer_to_training_kwargs_ranges: A mapping from model name to a mapping of trainer name
        to a mapping of keyword arguments for that trainer to be used in HPO.
    :param ablation_config: Additional third-order and fourth-order ablation configuration for all other ablation
        keys to models to either kwargs or kwarg ranges

    :param optuna_config: Configuration passed to optuna for HPO over all ablation studies
    :param directory: The directory in which the experimental artifacts will be saved.
    :param dry_run: Defines whether only the configurations for the single experiments should be created without
     running them.
    :param best_replicates: Defines how often the final model should be re-trained and evaluated based on the best
     hyper-parameters enabling to measure the variance in performance.
    :param save_artifacts: Defines, whether each trained model sampled during HPO should be saved.
    :param move_to_cpu: Defines, whether a replicate of the best model should be moved to CPU.
     We recommend to set this flag to 'True' to avoid unnecessary GPU usage.
    :param discard_replicates: Defines, whether the best model should be discarded after training and evaluation.
    """
    datetime = time.strftime('%Y-%m-%d-%H-%M')
    directory = os.path.join(directory, f'{datetime}_{uuid4()}')

    directories = prepare_ablation(
        datasets=datasets,
        create_inverse_triples=create_inverse_triples,
        models=models,
        model_to_model_kwargs=model_to_model_kwargs,
        model_to_model_kwargs_ranges=model_to_model_kwargs_ranges,
        model_to_trainer_to_training_kwargs=model_to_trainer_to_training_kwargs,
        model_to_trainer_to_training_kwargs_ranges=model_to_trainer_to_training_kwargs_ranges,
        losses=losses,
        regularizers=regularizers,
        optimizers=optimizers,
        training_loops=training_loops,
        evaluator=evaluator,
        optuna_config=optuna_config,
        ablation_config=ablation_config,
        evaluator_kwargs=evaluator_kwargs,
        evaluation_kwargs=evaluation_kwargs,
        directory=directory,
        save_artifacts=save_artifacts,
    )
    if dry_run:
        return

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


def ablation_pipeline_from_config(
    config: Mapping[str, Any],
    *,
    directory: Optional[str] = None,
    dry_run: bool = False,
    best_replicates: Optional[int] = None,
    save_artifacts: bool = True,
    move_to_cpu: bool = True,
    discard_replicates: bool = False,
) -> None:
    """Generate a set of HPO configurations.

    A sample file can be run with``pykeen experiment ablation tests/resources/hpo_complex_nations.json``.

    :param config: Dictionary defining the ablation studies.
    :param directory: The directory in which the experimental artifacts will be saved.
    :param dry_run: Defines whether only the configurations for the single experiments should be created without
     running them.
    :param best_replicates: Defines how often the final model should be re-trained and evaluated based on the best
     hyper-parameters enabling to measure the variance in performance.
    :param save_artifacts: Defines, whether each trained model sampled during HPO should be saved.
    :param move_to_cpu: Defines, whether a replicate of the best model should be moved to CPU.
     We recommend to set this flag to 'True' to avoid unnecessary GPU usage.
    :param discard_replicates: Defines, whether the best model should be discarded after training and evaluation.
    """
    return ablation_pipeline(
        **config,
        directory=directory,
        dry_run=dry_run,
        best_replicates=best_replicates,
        save_artifacts=save_artifacts,
        move_to_cpu=move_to_cpu,
        discard_replicates=discard_replicates,
    )


def prepare_ablation_from_path(path: str, directory: str, save_artifacts: bool) -> List[Tuple[str, str]]:
    """Prepare a set of ablation study directories.

    :param path: Path to configuration file defining the ablation studies.
    :param directory: The directory in which the experimental artifacts (including the ablation configurations)
     will be saved.
    :param save_artifacts: Defines, whether the output directories for the trained models sampled during HPO should be
     created.
    """
    with open(path) as file:
        config = json.load(file)
    return prepare_ablation_from_config(config=config, directory=directory, save_artifacts=save_artifacts)


def prepare_ablation_from_config(
    config: Mapping[str, Any],
    directory: str,
    save_artifacts: bool,
) -> List[Tuple[str, str]]:
    """Prepare a set of ablation study directories.

    :param config: Dictionary defining the ablation studies.
    :param directory: The directory in which the experimental artifacts (including the ablation configurations)
     will be saved.
    :param save_artifacts: Defines, whether the output directories for the trained models sampled during HPO should be
     created.

    :return: pairs of output directories and HPO config paths inside those directories
    """
    metadata = config['metadata']
    optuna_config = config['optuna']
    ablation_config = config['ablation']

    evaluator = ablation_config['evaluator']
    evaluator_kwargs = ablation_config['evaluator_kwargs']
    evaluation_kwargs = ablation_config['evaluation_kwargs']

    datasets = ablation_config['datasets']
    create_inverse_triples = ablation_config['create_inverse_triples']
    models = ablation_config['models']
    losses = ablation_config['loss_functions'] if 'loss_functions' in ablation_config else ablation_config['losses']
    regularizers = ablation_config['regularizers']
    optimizers = ablation_config['optimizers']
    training_loops = ablation_config['training_loops']
    return prepare_ablation(
        datasets=datasets,
        create_inverse_triples=create_inverse_triples,
        models=models,
        losses=losses,
        regularizers=regularizers,
        optimizers=optimizers,
        training_loops=training_loops,
        evaluator=evaluator,
        optuna_config=optuna_config,
        ablation_config=ablation_config,
        evaluator_kwargs=evaluator_kwargs,
        evaluation_kwargs=evaluation_kwargs,
        metadata=metadata,
        directory=directory,
        save_artifacts=save_artifacts,
    )


def prepare_ablation(  # noqa:C901
    datasets: Union[str, List[str]],
    models: Union[str, List[str]],
    losses: Union[str, List[str]],
    optimizers: Union[str, List[str]],
    training_loops: Union[str, List[str]],
    *,
    ablation_config: Optional[Mapping3D] = None,
    create_inverse_triples: Union[bool, List[bool]] = False,
    regularizers: Union[None, str, List[str]] = None,
    model_to_model_kwargs: Optional[Mapping2D] = None,
    model_to_model_kwargs_ranges: Optional[Mapping2D] = None,
    model_to_trainer_to_training_kwargs: Optional[Mapping3D] = None,
    model_to_trainer_to_training_kwargs_ranges: Optional[Mapping3D] = None,
    evaluator: Optional[str] = None,
    optuna_config: Optional[Mapping[str, Any]] = None,
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    metadata=None,
    directory: Optional[str] = None,
    save_artifacts: bool = True,
) -> List[Tuple[str, str]]:
    """Prepare an ablation directory.

    :return: pairs of output directories and HPO config paths inside those directories
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(create_inverse_triples, bool):
        create_inverse_triples = [create_inverse_triples]
    if isinstance(models, str):
        models = [models]
    if isinstance(losses, str):
        losses = [losses]
    if isinstance(optimizers, str):
        optimizers = [optimizers]
    if isinstance(training_loops, str):
        training_loops = [training_loops]
    if isinstance(regularizers, str) or regularizers is None:
        regularizers = [regularizers]

    it = itt.product(
        datasets,
        create_inverse_triples,
        models,
        losses,
        regularizers,
        optimizers,
        training_loops,
    )

    if not model_to_model_kwargs:
        model_to_model_kwargs = {}
    if not model_to_model_kwargs_ranges:
        model_to_model_kwargs_ranges = {}
    if not model_to_trainer_to_training_kwargs:
        model_to_trainer_to_training_kwargs = {}
    if not model_to_trainer_to_training_kwargs_ranges:
        model_to_trainer_to_training_kwargs_ranges = {}
    if not ablation_config:
        ablation_config = {}

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

        _experiment_optuna_config = optuna_config.copy() if optuna_config else {}
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
        #    random_seed=random_non_negative_int(),
        # ),

        def _set_arguments(key: str, value: str) -> None:
            """Set argument and its values."""
            d = {key: value}
            kwargs = ablation_config.get(f'{key}_kwargs', {}).get(model, {}).get(value, {})
            if kwargs:
                d[f'{key}_kwargs'] = kwargs
            kwargs_ranges = ablation_config.get(f'{key}_kwargs_ranges', {}).get(model, {}).get(value, {})
            if kwargs_ranges:
                d[f'{key}_kwargs_ranges'] = kwargs_ranges

            hpo_config.update(d)

        # Add dataset to current_pipeline
        if isinstance(dataset, str):
            hpo_config['dataset'] = dataset
        elif isinstance(dataset, dict):
            # Training, test, and validation paths are provided
            hpo_config['training'] = dataset['training']
            hpo_config['testing'] = dataset['testing']
            hpo_config['validation'] = dataset['validation']
        else:
            TypeError("Dataset must be either the dataset name, i.e., of type str, or a dictionary containing\n"
                      "the paths to the training, testing, and validation data.")
        logger.info(f"Dataset: {dataset}")
        hpo_config['dataset_kwargs'] = dict(create_inverse_triples=create_inverse_triples)
        logger.info(f"Add inverse triples: {create_inverse_triples}")

        hpo_config['model'] = model
        hpo_config['model_kwargs'] = model_to_model_kwargs.get(model, {})
        hpo_config['model_kwargs_ranges'] = model_to_model_kwargs_ranges.get(model, {})
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
            negative_sampler = ablation_config.get('negative_sampler', 'basic')  # default to basic
            _set_arguments(key='negative_sampler', value=negative_sampler)
            logger.info(f"Negative sampler: {negative_sampler}")

        # Add training kwargs and kwargs_ranges
        hpo_config['training_kwargs'] = model_to_trainer_to_training_kwargs.get(model, {}).get(training_loop, {})
        hpo_config['training_kwargs_ranges'] = model_to_trainer_to_training_kwargs_ranges.get(model, {}).get(
            training_loop, {})

        # Add evaluation
        hpo_config['evaluator'] = evaluator
        if evaluator_kwargs:
            hpo_config['evaluator_kwargs'] = evaluator_kwargs
        hpo_config['evaluation_kwargs'] = evaluation_kwargs or {}
        logger.info(f"Evaluator: {evaluator}")

        rv_config = dict(
            type='hpo',
            metadata=metadata or {},
            pipeline=hpo_config,
            optuna=_experiment_optuna_config,
        )

        rv_config_path = os.path.join(output_directory, 'hpo_config.json')
        with open(rv_config_path, 'w') as file:
            json.dump(rv_config, file, indent=2, ensure_ascii=True)

        directories.append((output_directory, rv_config_path))

    return directories
