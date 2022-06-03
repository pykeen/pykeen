# -*- coding: utf-8 -*-

"""Utilities for ablation study configurations."""

import itertools as itt
import json
import logging
import pathlib
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4

from ..training import SLCWATrainingLoop, training_loop_resolver
from ..utils import normalize_path, normalize_string

__all__ = [
    "ablation_pipeline",
    "ablation_pipeline_from_config",
    "prepare_ablation_from_config",
    "prepare_ablation_from_path",
    "prepare_ablation",
]

logger = logging.getLogger(__name__)

Mapping2D = Mapping[str, Mapping[str, Any]]
Mapping3D = Mapping[str, Mapping[str, Mapping[str, Any]]]


def ablation_pipeline(
    datasets: Union[str, List[str]],
    directory: Union[str, pathlib.Path],
    models: Union[str, List[str]],
    losses: Union[str, List[str]],
    optimizers: Union[str, List[str]],
    training_loops: Union[str, List[str]],
    *,
    epochs: Optional[int] = None,
    create_inverse_triples: Union[bool, List[bool]] = False,
    regularizers: Union[None, str, List[str]] = None,
    negative_sampler: Union[str, None] = None,
    evaluator: Optional[str] = None,
    stopper: Optional[str] = "NopStopper",
    model_to_model_kwargs: Optional[Mapping2D] = None,
    model_to_model_kwargs_ranges: Optional[Mapping2D] = None,
    model_to_loss_to_loss_kwargs: Optional[Mapping3D] = None,
    model_to_loss_to_loss_kwargs_ranges: Optional[Mapping3D] = None,
    model_to_optimizer_to_optimizer_kwargs: Optional[Mapping3D] = None,
    model_to_optimizer_to_optimizer_kwargs_ranges: Optional[Mapping3D] = None,
    model_to_negative_sampler_to_negative_sampler_kwargs: Optional[Mapping3D] = None,
    model_to_negative_sampler_to_negative_sampler_kwargs_ranges: Optional[Mapping3D] = None,
    model_to_training_loop_to_training_loop_kwargs: Optional[Mapping3D] = None,
    model_to_training_loop_to_training_kwargs: Optional[Mapping3D] = None,
    model_to_training_loop_to_training_kwargs_ranges: Optional[Mapping3D] = None,
    model_to_regularizer_to_regularizer_kwargs: Optional[Mapping3D] = None,
    model_to_regularizer_to_regularizer_kwargs_ranges: Optional[Mapping3D] = None,
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    stopper_kwargs: Optional[Mapping[str, Any]] = None,
    n_trials: Optional[int] = 5,
    timeout: Optional[int] = 3600,
    metric: Optional[str] = "hits@10",
    direction: Optional[str] = "maximize",
    sampler: Optional[str] = "random",
    pruner: Optional[str] = "nop",
    metadata: Optional[Mapping] = None,
    save_artifacts: bool = True,
    move_to_cpu: bool = True,
    dry_run: bool = False,
    best_replicates: Optional[int] = None,
    discard_replicates: bool = False,
    create_unique_subdir: bool = False,
):
    """Run ablation study.

    :param datasets: A dataset name or list of dataset names.
    :param directory: The directory in which the experimental artifacts will be saved.
    :param models: A model name or list of model names.
    :param losses: A loss function name or list of loss function names.
    :param optimizers: An optimizer name or list of optimizer names.
    :param training_loops: A training loop name or list of training loop names.
    :param epochs: A quick way to set the ``num_epochs`` in the training kwargs.
    :param create_inverse_triples: Either a boolean for a single entry or a list of booleans.
    :param regularizers: A regularizer name, list of regularizer names, or None if no regularizer is desired.
    :param negative_sampler: A negative sampler name, list of regularizer names, or None if no negative sampler
        is desired. Negative sampling is used only in combination with :class:`pykeen.training.SLCWATrainingLoop`.
    :param evaluator: The name of the evaluator to be used. Defaults to rank-based evaluator.
    :param stopper: The name of the stopper to be used. Defaults to NopStopper which doesn't define a
        stopping criterion.
    :param model_to_model_kwargs: A mapping from model name to dictionaries of default keyword arguments for
        the instantiation of that model.
    :param model_to_model_kwargs_ranges: A mapping from model name to dictionaries of keyword argument
        ranges for that model to be used in HPO.
    :param model_to_loss_to_loss_kwargs: A mapping from model name to a mapping of loss name to a mapping
        of default keyword arguments for the instantiation of that loss function. This is useful because for some
        losses, have hyper-parameters such as :class:`pykeen.losses.MarginRankingLoss`.
    :param model_to_loss_to_loss_kwargs_ranges: A mapping from model name to a mapping of loss name
        to a mapping of keyword argument ranges for that loss to be used in HPO.
    :param model_to_optimizer_to_optimizer_kwargs: A mapping from model name to a mapping of optimizer name to a mapping
        of default keyword arguments for the instantiation of that optimizer. This is useful because the optimizers,
        have hyper-parameters such as the learning rate.
    :param model_to_optimizer_to_optimizer_kwargs_ranges: A mapping from model name to a mapping of optimizer name
        to a mapping of keyword argument ranges for that optimizer to be used in HPO.
    :param model_to_regularizer_to_regularizer_kwargs: A mapping from model name to a mapping of regularizer name to a
        mapping of default keyword arguments for the instantiation of that regularizer. This is useful because the
        optimizers, have hyper-parameters such as the regularization weight.
    :param model_to_regularizer_to_regularizer_kwargs_ranges: A mapping from model name to a mapping of regularizer name
        to a mapping of keyword argument ranges for that regularizer to be used in HPO.
    :param model_to_negative_sampler_to_negative_sampler_kwargs: A mapping from model name to a mapping of
        negative sampler name to a mapping of default keyword arguments for the instantiation of that negative sampler.
        This is useful because the negative samplers, have hyper-parameters such as the number of negatives that should
        get generated for each positive training example.
    :param model_to_negative_sampler_to_negative_sampler_kwargs_ranges: A mapping from model name to a mapping of
        negative sampler name to a mapping of keyword argument ranges for that negative sampler to be used in HPO.
    :param model_to_training_loop_to_training_loop_kwargs: A mapping from model name to a mapping of training loop name
        to a mapping of default keyword arguments for the training loop.
    :param model_to_training_loop_to_training_kwargs: A mapping from model name to a mapping of trainer name to a
        mapping of default keyword arguments for the training procedure. This is useful because you can set the
        hyper-parameters such as the number of training epochs and the batch size.
    :param model_to_training_loop_to_training_kwargs_ranges:  A mapping from model name to a mapping of
        trainer name to a mapping of keyword argument ranges for that trainer to be used in HPO.
    :param evaluator_kwargs: The keyword arguments passed to the evaluator.
    :param evaluation_kwargs: The keyword arguments passed during evaluation.
    :param stopper_kwargs: The keyword arguments passed to the stopper.
    :param n_trials: Number of HPO trials.
    :param timeout: The time (seconds) after which the ablation study will be terminated.
    :param metric: The metric to optimize during HPO.
    :param direction: Defines, whether to 'maximize' or 'minimize' the metric during HPO.
    :param sampler: The HPO sampler, it defaults to random search.
    :param pruner: Defines approach for pruning trials. Per default no pruning is used, i.e., pruner is
        set to 'Nopruner'.
    :param metadata: A mapping of meta data arguments such as name of the ablation study.
    :param save_artifacts: Defines, whether each trained model sampled during HPO should be saved.
    :param move_to_cpu: Defines, whether a replicate of the best model should be moved to CPU.
    :param dry_run: Defines whether only the configurations for the single experiments should be created without
        running them.
    :param best_replicates: Defines how often the final model should be re-trained and evaluated based on the best
        hyper-parameters enabling to measure the variance in performance.
    :param discard_replicates: Defines, whether the best model should be discarded after training and evaluation.
    :param create_unique_subdir: Defines, whether a unique sub-directory for the experimental artifacts should
        be created. The sub-directory name is defined  by the  current  data + a unique id.
    """
    directory = normalize_path(directory, *iter_unique_ids(disable=not create_unique_subdir))
    directories = prepare_ablation(
        datasets=datasets,
        models=models,
        losses=losses,
        optimizers=optimizers,
        training_loops=training_loops,
        epochs=epochs,
        create_inverse_triples=create_inverse_triples,
        regularizers=regularizers,
        model_to_model_kwargs=model_to_model_kwargs,
        model_to_model_kwargs_ranges=model_to_model_kwargs_ranges,
        model_to_loss_to_loss_kwargs=model_to_loss_to_loss_kwargs,
        model_to_loss_to_loss_kwargs_ranges=model_to_loss_to_loss_kwargs_ranges,
        model_to_optimizer_to_optimizer_kwargs=model_to_optimizer_to_optimizer_kwargs,
        model_to_optimizer_to_optimizer_kwargs_ranges=model_to_optimizer_to_optimizer_kwargs_ranges,
        negative_sampler=negative_sampler,
        model_to_neg_sampler_to_neg_sampler_kwargs=model_to_negative_sampler_to_negative_sampler_kwargs,
        model_to_neg_sampler_to_neg_sampler_kwargs_ranges=model_to_negative_sampler_to_negative_sampler_kwargs_ranges,
        model_to_training_loop_to_training_loop_kwargs=model_to_training_loop_to_training_loop_kwargs,
        model_to_training_loop_to_training_kwargs=model_to_training_loop_to_training_kwargs,
        model_to_training_loop_to_training_kwargs_ranges=model_to_training_loop_to_training_kwargs_ranges,
        model_to_regularizer_to_regularizer_kwargs=model_to_regularizer_to_regularizer_kwargs,
        model_to_regularizer_to_regularizer_kwargs_ranges=model_to_regularizer_to_regularizer_kwargs_ranges,
        evaluator=evaluator,
        n_trials=n_trials,
        timeout=timeout,
        metric=metric,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        evaluator_kwargs=evaluator_kwargs,
        evaluation_kwargs=evaluation_kwargs,
        stopper=stopper,
        stopper_kwargs=stopper_kwargs,
        metadata=metadata,
        directory=directory,
        save_artifacts=save_artifacts,
    )

    _run_ablation_experiments(
        directories=directories,
        best_replicates=best_replicates,
        dry_run=dry_run,
        move_to_cpu=move_to_cpu,
        discard_replicates=discard_replicates,
    )


def _run_ablation_experiments(
    directories: Sequence[Tuple[Union[str, pathlib.Path], Union[str, pathlib.Path]]],
    best_replicates: Optional[int] = None,
    dry_run: bool = False,
    move_to_cpu: bool = True,
    discard_replicates: bool = False,
) -> None:
    """Run ablation experiments."""
    if dry_run:
        return

    from pykeen.hpo import hpo_pipeline_from_path

    for output_directory, rv_config_path in directories:
        if isinstance(output_directory, str):
            output_directory = pathlib.Path(output_directory).resolve()
        hpo_pipeline_result = hpo_pipeline_from_path(rv_config_path)
        hpo_pipeline_result.save_to_directory(output_directory)

        if not best_replicates:
            continue

        best_pipeline_dir = output_directory.joinpath("best_pipeline")
        best_pipeline_dir.mkdir(exist_ok=True, parents=True)
        logger.info("Re-training best pipeline and saving artifacts in %s", best_pipeline_dir)
        hpo_pipeline_result.replicate_best_pipeline(
            replicates=best_replicates,
            move_to_cpu=move_to_cpu,
            save_replicates=not discard_replicates,
            directory=best_pipeline_dir,
        )


def iter_unique_ids(disable: bool = False) -> Iterable[str]:
    """Iterate unique id to append to a path."""
    if disable:
        return []
    datetime = time.strftime("%Y-%m-%d-%H-%M")
    yield f"{datetime}_{uuid4()}"


def ablation_pipeline_from_config(
    config: Mapping[str, Any],
    directory: str,
    *,
    dry_run: bool = False,
    best_replicates: Optional[int] = None,
    save_artifacts: bool = True,
    move_to_cpu: bool = True,
    discard_replicates: bool = False,
):
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
    :return: None.
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


def prepare_ablation_from_path(
    path: Union[str, pathlib.Path],
    directory: Union[str, pathlib.Path],
    save_artifacts: bool,
) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """Prepare a set of ablation study directories.

    :param path: Path to configuration file defining the ablation studies.
    :param directory: The directory in which the experimental artifacts (including the ablation configurations)
        will be saved.
    :param save_artifacts: Defines, whether the output directories for the trained models sampled during HPO should be
        created.
    :return: pairs of output directories and HPO config paths inside those directories
    """
    directory = normalize_path(directory, *iter_unique_ids())
    with open(path) as file:
        config = json.load(file)
    return prepare_ablation_from_config(config=config, directory=directory, save_artifacts=save_artifacts)


def prepare_ablation_from_config(
    config: Mapping[str, Any],
    directory: Union[str, pathlib.Path],
    save_artifacts: bool,
) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """Prepare a set of ablation study directories.

    :param config: Dictionary defining the ablation studies.
    :param directory: The directory in which the experimental artifacts (including the ablation configurations)
        will be saved.
    :param save_artifacts: Defines, whether the output directories for the trained models sampled during HPO should be
        created.

    :return: pairs of output directories and HPO config paths inside those directories
    """
    metadata = config["metadata"]
    optuna_config = config["optuna"]
    ablation_config = config["ablation"]

    return prepare_ablation(
        **ablation_config,
        **optuna_config,
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
    directory: Union[str, pathlib.Path],
    *,
    epochs: Optional[int] = None,
    create_inverse_triples: Union[bool, List[bool]] = False,
    regularizers: Union[None, str, List[str], List[None]] = None,
    negative_sampler: Optional[str] = None,
    evaluator: Optional[str] = None,
    model_to_model_kwargs: Optional[Mapping2D] = None,
    model_to_model_kwargs_ranges: Optional[Mapping2D] = None,
    model_to_loss_to_loss_kwargs: Optional[Mapping3D] = None,
    model_to_loss_to_loss_kwargs_ranges: Optional[Mapping3D] = None,
    model_to_optimizer_to_optimizer_kwargs: Optional[Mapping3D] = None,
    model_to_optimizer_to_optimizer_kwargs_ranges: Optional[Mapping3D] = None,
    model_to_training_loop_to_training_loop_kwargs: Optional[Mapping3D] = None,
    model_to_neg_sampler_to_neg_sampler_kwargs: Optional[Mapping3D] = None,
    model_to_neg_sampler_to_neg_sampler_kwargs_ranges: Optional[Mapping3D] = None,
    model_to_training_loop_to_training_kwargs: Optional[Mapping3D] = None,
    model_to_training_loop_to_training_kwargs_ranges: Optional[Mapping3D] = None,
    model_to_regularizer_to_regularizer_kwargs: Optional[Mapping3D] = None,
    model_to_regularizer_to_regularizer_kwargs_ranges: Optional[Mapping3D] = None,
    n_trials: Optional[int] = 5,
    timeout: Optional[int] = 3600,
    metric: Optional[str] = "hits@10",
    direction: Optional[str] = "maximize",
    sampler: Optional[str] = "random",
    pruner: Optional[str] = "nop",
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    stopper: Optional[str] = "NopStopper",
    stopper_kwargs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping] = None,
    save_artifacts: bool = True,
) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """Prepare an ablation directory.

    :param datasets: A dataset name or list of dataset names.
    :param models: A model name or list of model names.
    :param losses: A loss function name or list of loss function names.
    :param optimizers: An optimizer name or list of optimizer names.
    :param training_loops: A training loop name or list of training loop names.
    :param epochs: A quick way to set the ``num_epochs`` in the training kwargs.
    :param create_inverse_triples: Either a boolean for a single entry or a list of booleans.
    :param regularizers: A regularizer name, list of regularizer names, or None if no regularizer is desired.
    :param negative_sampler: A negative sampler name, list of regularizer names, or None if no negative sampler
        is desired. Negative sampling is used only in combination with the pykeen.training.sclwa training loop.
    :param evaluator: The name of the evaluator to be used. Defaults to rank-based evaluator.
    :param stopper: The name of the stopper to be used. Defaults to NopStopper which doesn't define a
        stopping criterion.
    :param model_to_model_kwargs: A mapping from model name to dictionaries of default keyword arguments for
        the instantiation of that model.
    :param model_to_model_kwargs_ranges: A mapping from model name to dictionaries of keyword argument
        ranges for that model to be used in HPO.
    :param model_to_loss_to_loss_kwargs: A mapping from model name to a mapping of loss name to a mapping
        of default keyword arguments for the instantiation of that loss function. This is useful because for some
        losses, have hyper-parameters such as pykeen.losses.MarginRankingLoss
    :param model_to_loss_to_loss_kwargs_ranges: A mapping from model name to a mapping of loss name
        to a mapping of keyword argument ranges for that loss to be used in HPO.
    :param model_to_optimizer_to_optimizer_kwargs: A mapping from model name to a mapping of optimizer name to a mapping
        of default keyword arguments for the instantiation of that optimizer. This is useful because the optimizers,
        have hyper-parameters such as the learning rate.
    :param model_to_optimizer_to_optimizer_kwargs_ranges: A mapping from model name to a mapping of optimizer name
        to a mapping of keyword argument ranges for that optimizer to be used in HPO.
    :param model_to_regularizer_to_regularizer_kwargs: A mapping from model name to a mapping of regularizer name to a
        mapping of default keyword arguments for the instantiation of that regularizer. This is useful because the
        optimizers, have hyper-parameters such as the regularization weight.
    :param model_to_regularizer_to_regularizer_kwargs_ranges: A mapping from model name to a mapping of regularizer name
        to a mapping of keyword argument ranges for that regularizer to be used in HPO.
    :param model_to_neg_sampler_to_neg_sampler_kwargs: A mapping from model name to a mapping of
        negative sampler name to a mapping of default keyword arguments for the instantiation of that negative sampler.
        This is useful because the negative samplers, have hyper-parameters such as the number of negatives that should
        get generated for each positive training example.
    :param model_to_neg_sampler_to_neg_sampler_kwargs_ranges: A mapping from model name to a mapping of
        negative sampler name to a mapping of keyword argument ranges for that negative sampler to be used in HPO.
    :param model_to_training_loop_to_training_loop_kwargs: A mapping from model name to a mapping of training loop name
        to a mapping of default keyword arguments for the training loop.
    :param model_to_training_loop_to_training_kwargs: A mapping from model name to a mapping of trainer name to a
        mapping of default keyword arguments for the training procedure. This is useful because you can set the
        hyper-parameters such as the number of training epochs and the batch size.
    :param model_to_training_loop_to_training_kwargs_ranges:  A mapping from model name to a mapping of
        trainer name to a mapping of keyword argument ranges for that trainer to be used in HPO.
    :param evaluator_kwargs: The keyword arguments passed to the evaluator.
    :param evaluation_kwargs: The keyword arguments passed during evaluation.
    :param stopper_kwargs: The keyword arguments passed to the stopper.
    :param n_trials: Number of HPO trials.
    :param timeout: The time (seconds) after which the ablation study will be terminated.
    :param metric: The metric to optimize during HPO.
    :param direction: Defines, whether to 'maximize' or 'minimize' the metric during HPO.
    :param sampler: The HPO sampler, it defaults to random search.
    :param pruner: Defines approach for pruning trials. Per default no pruning is used, i.e., pruner is
        set to 'Nopruner'.
    :param metadata: A mapping of meta data arguments such as name of the ablation study.
    :param directory: The directory in which the experimental artifacts will be saved.
    :param save_artifacts: Defines, whether each trained model sampled during HPO should be saved.

    :return: pairs of output directories and HPO config paths inside those directories.
    :raises ValueError:
            If the dataset is not specified correctly, i.e., dataset is not of type str, or a dictionary containing
            the paths to the training, testing, and validation data.
    """
    directory = normalize_path(path=directory)
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
    if isinstance(regularizers, str):
        regularizers = [regularizers]
    elif regularizers is None:
        regularizers = [None]

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
        dataset_name = normalize_string(dataset) if isinstance(dataset, str) else "user_data"
        experiment_name = f"{counter:04d}_{dataset_name}_{normalize_string(model)}"
        output_directory = directory.joinpath(experiment_name)
        output_directory.mkdir(exist_ok=True, parents=True)
        # TODO what happens if already exists?

        _experiment_optuna_config = {
            "n_trials": n_trials,
            "timeout": timeout,
            "metric": metric,
            "direction": direction,
            "sampler": sampler,
            "pruner": pruner,
        }
        _experiment_optuna_config["storage"] = f"sqlite:///{output_directory.as_posix()}/optuna_results.db"
        if save_artifacts:
            save_model_directory = output_directory.joinpath("artifacts")
            save_model_directory.mkdir(exist_ok=True, parents=True)
            _experiment_optuna_config["save_model_directory"] = save_model_directory.as_posix()

        hpo_config: Dict[str, Any] = dict()
        hpo_config["stopper"] = stopper

        if stopper_kwargs is not None:
            hpo_config["stopper_kwargs"] = stopper_kwargs

        # TODO incorporate setting of random seed
        # pipeline_kwargs=dict(
        #    random_seed=random_non_negative_int(),
        # ),

        def _set_arguments(config: Optional[Mapping3D], key: str, value: str) -> None:
            """Set argument and its values."""
            d = {}
            d[key] = {} if config is None else config.get(model, {}).get(value, {})
            if d[key]:
                hpo_config.update(d)

        # Add dataset to current_pipeline
        if isinstance(dataset, str):
            hpo_config["dataset"] = dataset
        elif isinstance(dataset, dict):
            # Training, test, and validation paths are provided
            hpo_config["training"] = dataset["training"]
            hpo_config["testing"] = dataset["testing"]
            hpo_config["validation"] = dataset["validation"]
        else:
            raise ValueError(
                "Dataset must be either the dataset name, i.e., of type str, or a dictionary containing\n"
                "the paths to the training, testing, and validation data.",
            )
        logger.info(f"Dataset: {dataset}")
        hpo_config["dataset_kwargs"] = dict(create_inverse_triples=create_inverse_triples)
        logger.info(f"Add inverse triples: {create_inverse_triples}")

        hpo_config["model"] = model
        hpo_config["model_kwargs"] = model_to_model_kwargs.get(model, {})
        hpo_config["model_kwargs_ranges"] = model_to_model_kwargs_ranges.get(model, {})
        logger.info(f"Model: {model}")

        # Add loss function to current_pipeline
        hpo_config["loss"] = loss
        _set_arguments(config=model_to_loss_to_loss_kwargs, key="loss_kwargs", value=loss)
        _set_arguments(config=model_to_loss_to_loss_kwargs_ranges, key="loss_kwargs_ranges", value=loss)
        logger.info(f"Loss functions: {loss}")

        # Add regularizer to current_pipeline
        if regularizer is not None:
            hpo_config["regularizer"] = regularizer
            _set_arguments(
                config=model_to_regularizer_to_regularizer_kwargs,
                key="regularizer_kwargs",
                value=regularizer,
            )
            _set_arguments(
                config=model_to_regularizer_to_regularizer_kwargs_ranges,
                key="regularizer_kwargs_ranges",
                value=regularizer,
            )
            logger.info(f"Regularizer: {regularizer}")

        # Add optimizer to current_pipeline
        hpo_config["optimizer"] = optimizer
        _set_arguments(config=model_to_optimizer_to_optimizer_kwargs, key="optimizer_kwargs", value=optimizer)
        _set_arguments(
            config=model_to_optimizer_to_optimizer_kwargs_ranges,
            key="optimizer_kwargs_ranges",
            value=optimizer,
        )
        logger.info(f"Optimizer: {optimizer}")

        # Add training approach to current_pipeline
        hpo_config["training_loop"] = training_loop
        _set_arguments(
            config=model_to_training_loop_to_training_loop_kwargs,
            key="training_loop_kwargs",
            value=training_loop,
        )
        _set_arguments(config=model_to_training_loop_to_training_kwargs, key="training_kwargs", value=training_loop)
        _set_arguments(
            config=model_to_training_loop_to_training_kwargs_ranges,
            key="training_kwargs_ranges",
            value=training_loop,
        )
        logger.info(f"Training loop: {training_loop}")

        if issubclass(training_loop_resolver.lookup(training_loop), SLCWATrainingLoop):
            negative_sampler = negative_sampler or "basic"  # default to basic
            _set_arguments(
                config=model_to_neg_sampler_to_neg_sampler_kwargs,
                key="negative_sampler_kwargs",
                value=negative_sampler,
            )
            _set_arguments(
                config=model_to_neg_sampler_to_neg_sampler_kwargs_ranges,
                key="negative_sampler_kwargs_ranges",
                value=negative_sampler,
            )
            logger.info(f"Negative sampler: {negative_sampler}")

        # Add evaluation
        hpo_config["evaluator"] = evaluator
        if evaluator_kwargs:
            hpo_config["evaluator_kwargs"] = evaluator_kwargs
        hpo_config["evaluation_kwargs"] = evaluation_kwargs or {}
        logger.info(f"Evaluator: {evaluator}")

        if epochs is not None:
            hpo_config.setdefault("training_kwargs", {}).setdefault("num_epochs", epochs)

        rv_config = dict(
            type="hpo",
            metadata=metadata or {},
            pipeline=hpo_config,
            optuna=_experiment_optuna_config,
        )

        rv_config_path = output_directory.joinpath("hpo_config.json")
        with rv_config_path.open("w") as file:
            json.dump(rv_config, file, indent=2, ensure_ascii=True)

        directories.append((output_directory, rv_config_path))

    return directories
