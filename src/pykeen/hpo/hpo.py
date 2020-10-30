# -*- coding: utf-8 -*-

"""Hyper-parameter optimiziation in PyKEEN."""

import dataclasses
import ftplib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Collection, Dict, Mapping, Optional, Type, Union

import torch
from optuna import Study, Trial, create_study
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage

from .pruners import get_pruner_cls
from .samplers import get_sampler_cls
from ..datasets import get_dataset, has_dataset
from ..datasets.base import DataSet
from ..evaluation import Evaluator, get_evaluator_cls
from ..losses import Loss, _LOSS_SUFFIX, get_loss_cls
from ..models import get_model_cls
from ..models.base import Model
from ..optimizers import Optimizer, get_optimizer_cls, optimizers_hpo_defaults
from ..pipeline import pipeline, replicate_pipeline_from_config
from ..regularizers import Regularizer, get_regularizer_cls
from ..sampling import NegativeSampler, get_negative_sampler_cls
from ..stoppers import EarlyStopper, Stopper, get_stopper_cls
from ..trackers import ResultTracker, get_result_tracker_cls
from ..training import SLCWATrainingLoop, TrainingLoop, get_training_loop_cls
from ..triples import TriplesFactory
from ..utils import (
    Result, ensure_ftp_directory, fix_dataclass_init_docs, get_df_io, get_json_bytes_io,
    normalize_string,
)
from ..version import get_git_hash, get_version

__all__ = [
    'hpo_pipeline_from_path',
    'hpo_pipeline_from_config',
    'hpo_pipeline',
    'HpoPipelineResult',
]

logger = logging.getLogger(__name__)

STOPPED_EPOCH_KEY = 'stopped_epoch'


@dataclass
class Objective:
    """A dataclass containing all of the information to make an objective function."""

    dataset: Union[None, str, Type[DataSet]]  # 1.
    model: Type[Model]  # 2.
    loss: Type[Loss]  # 3.
    regularizer: Type[Regularizer]  # 4.
    optimizer: Type[Optimizer]  # 5.
    training_loop: Type[TrainingLoop]  # 6.
    evaluator: Type[Evaluator]  # 8.
    result_tracker: Type[ResultTracker]  # 9.

    # 1. Dataset
    dataset_kwargs: Optional[Mapping[str, Any]] = None
    training: Union[None, TriplesFactory, str] = None
    testing: Union[None, TriplesFactory, str] = None
    validation: Union[None, TriplesFactory, str] = None
    evaluation_entity_whitelist: Optional[Collection[str]] = None
    evaluation_relation_whitelist: Optional[Collection[str]] = None
    # 2. Model
    model_kwargs: Optional[Mapping[str, Any]] = None
    model_kwargs_ranges: Optional[Mapping[str, Any]] = None
    # 3. Loss
    loss_kwargs: Optional[Mapping[str, Any]] = None
    loss_kwargs_ranges: Optional[Mapping[str, Any]] = None
    # 4. Regularizer
    regularizer_kwargs: Optional[Mapping[str, Any]] = None
    regularizer_kwargs_ranges: Optional[Mapping[str, Any]] = None
    # 5. Optimizer
    optimizer_kwargs: Optional[Mapping[str, Any]] = None
    optimizer_kwargs_ranges: Optional[Mapping[str, Any]] = None
    # 6. Training Loop
    negative_sampler: Optional[Type[NegativeSampler]] = None
    negative_sampler_kwargs: Optional[Mapping[str, Any]] = None
    negative_sampler_kwargs_ranges: Optional[Mapping[str, Any]] = None
    # 7. Training
    training_kwargs: Optional[Mapping[str, Any]] = None
    training_kwargs_ranges: Optional[Mapping[str, Any]] = None
    stopper: Type[Stopper] = None
    stopper_kwargs: Optional[Mapping[str, Any]] = None
    # 8. Evaluation
    evaluator_kwargs: Optional[Mapping[str, Any]] = None
    evaluation_kwargs: Optional[Mapping[str, Any]] = None
    # 9. Trackers
    result_tracker_kwargs: Optional[Mapping[str, Any]] = None
    # Misc.
    metric: str = None
    device: Union[None, str, torch.device] = None
    save_model_directory: Optional[str] = None

    @staticmethod
    def _update_stopper_callbacks(stopper_kwargs: Dict[str, Any], trial: Trial) -> None:
        """Make a subclass of the EarlyStopper that reports to the trial."""

        def _result_callback(_early_stopper: EarlyStopper, result: Union[float, int], epoch: int) -> None:
            trial.report(result, step=epoch)

        def _stopped_callback(_early_stopper: EarlyStopper, _result: Union[float, int], epoch: int) -> None:
            trial.set_user_attr(STOPPED_EPOCH_KEY, epoch)

        for key, callback in zip(('result_callbacks', 'stopped_callbacks'), (_result_callback, _stopped_callback)):
            stopper_kwargs.setdefault(key, []).append(callback)

    def __call__(self, trial: Trial) -> Optional[float]:
        """Suggest parameters then train the model."""
        if self.model_kwargs is not None:
            problems = [
                x
                for x in ('loss', 'regularizer', 'optimizer', 'training', 'negative_sampler', 'stopper')
                if x in self.model_kwargs
            ]
            if problems:
                raise ValueError(f'model_kwargs should not have: {problems}. {self}')

        # 2. Model
        _model_kwargs = _get_kwargs(
            trial=trial,
            prefix='model',
            default_kwargs_ranges=self.model.hpo_default,
            kwargs=self.model_kwargs,
            kwargs_ranges=self.model_kwargs_ranges,
        )

        try:
            loss_default_kwargs_ranges = self.loss.hpo_default
        except AttributeError:
            logger.warning('using a loss function with no hpo_default field: %s', self.loss)
            loss_default_kwargs_ranges = {}

        # 3. Loss
        _loss_kwargs = _get_kwargs(
            trial=trial,
            prefix='loss',
            default_kwargs_ranges=loss_default_kwargs_ranges,
            kwargs=self.loss_kwargs,
            kwargs_ranges=self.loss_kwargs_ranges,
        )
        # 4. Regularizer
        _regularizer_kwargs = _get_kwargs(
            trial=trial,
            prefix='regularizer',
            default_kwargs_ranges=self.regularizer.hpo_default,
            kwargs=self.regularizer_kwargs,
            kwargs_ranges=self.regularizer_kwargs_ranges,
        )
        # 5. Optimizer
        _optimizer_kwargs = _get_kwargs(
            trial=trial,
            prefix='optimizer',
            default_kwargs_ranges=optimizers_hpo_defaults[self.optimizer],
            kwargs=self.optimizer_kwargs,
            kwargs_ranges=self.optimizer_kwargs_ranges,
        )

        if self.training_loop is not SLCWATrainingLoop:
            _negative_sampler_kwargs = {}
        else:
            _negative_sampler_kwargs = _get_kwargs(
                trial=trial,
                prefix='negative_sampler',
                default_kwargs_ranges=self.negative_sampler.hpo_default,
                kwargs=self.negative_sampler_kwargs,
                kwargs_ranges=self.negative_sampler_kwargs_ranges,
            )

        _training_kwargs = _get_kwargs(
            trial=trial,
            prefix='training',
            default_kwargs_ranges=self.training_loop.hpo_default,
            kwargs=self.training_kwargs,
            kwargs_ranges=self.training_kwargs_ranges,
        )

        _stopper_kwargs = dict(self.stopper_kwargs or {})
        if self.stopper is not None and issubclass(self.stopper, EarlyStopper):
            self._update_stopper_callbacks(_stopper_kwargs, trial)

        try:
            result = pipeline(
                # 1. Dataset
                dataset=self.dataset,
                dataset_kwargs=self.dataset_kwargs,
                training=self.training,
                testing=self.testing,
                validation=self.validation,
                evaluation_entity_whitelist=self.evaluation_entity_whitelist,
                evaluation_relation_whitelist=self.evaluation_relation_whitelist,
                # 2. Model
                model=self.model,
                model_kwargs=_model_kwargs,
                # 3. Loss
                loss=self.loss,
                loss_kwargs=_loss_kwargs,
                # 4. Regularizer
                regularizer=self.regularizer,
                regularizer_kwargs=_regularizer_kwargs,
                clear_optimizer=True,
                # 5. Optimizer
                optimizer=self.optimizer,
                optimizer_kwargs=_optimizer_kwargs,
                # 6. Training Loop
                training_loop=self.training_loop,
                negative_sampler=self.negative_sampler,
                negative_sampler_kwargs=_negative_sampler_kwargs,
                # 7. Training
                training_kwargs=_training_kwargs,
                stopper=self.stopper,
                stopper_kwargs=_stopper_kwargs,
                # 8. Evaluation
                evaluator=self.evaluator,
                evaluator_kwargs=self.evaluator_kwargs,
                evaluation_kwargs=self.evaluation_kwargs,
                # 9. Tracker
                result_tracker=self.result_tracker,
                result_tracker_kwargs=self.result_tracker_kwargs,
                # Misc.
                use_testing_data=False,  # use validation set during HPO!
                device=self.device,
            )
        except (MemoryError, RuntimeError) as e:
            trial.set_user_attr('failure', str(e))
            # Will trigger Optuna to set the state of the trial as failed
            return None
        else:
            if self.save_model_directory:
                model_directory = os.path.join(self.save_model_directory, str(trial.number))
                os.makedirs(model_directory, exist_ok=True)
                result.save_to_directory(model_directory)

            trial.set_user_attr('random_seed', result.random_seed)

            for k, v in result.metric_results.to_flat_dict().items():
                trial.set_user_attr(k, v)

            return result.metric_results.get_metric(self.metric)


@fix_dataclass_init_docs
@dataclass
class HpoPipelineResult(Result):
    """A container for the results of the HPO pipeline."""

    #: The :mod:`optuna` study object
    study: Study
    #: The objective class, containing information on preset hyper-parameters and those to optimize
    objective: Objective

    def _get_best_study_config(self):
        metadata = {
            'best_trial_number': self.study.best_trial.number,
            'best_trial_evaluation': self.study.best_value,
        }

        pipeline_config = dict()
        for k, v in self.study.user_attrs.items():
            if k.startswith('pykeen_'):
                metadata[k[len('pykeen_'):]] = v
            elif k in {'metric'}:
                continue
            else:
                pipeline_config[k] = v

        for field in dataclasses.fields(self.objective):
            if (not field.name.endswith('_kwargs') and field.name not in {
                'training',
                'testing',
                'validation',
            }) or field.name in {'metric'}:
                continue
            field_kwargs = getattr(self.objective, field.name)
            if field_kwargs:
                logger.debug(f'saving pre-specified field in pipeline config: {field.name}={field_kwargs}')
                pipeline_config[field.name] = field_kwargs

        for k, v in self.study.best_params.items():
            sk, ssk = k.split('.')
            sk = f'{sk}_kwargs'
            if sk not in pipeline_config:
                pipeline_config[sk] = {}
            logger.debug(f'saving optimized field in pipeline config: {sk}.{ssk}={v}')
            pipeline_config[sk][ssk] = v

        for k in ('stopper', 'stopper_kwargs'):
            if k in pipeline_config:
                v = pipeline_config.pop(k)
                metadata[f'_{k}_removed_comment'] = f'{k} config removed after HPO: {v}'

        stopped_epoch = self.study.best_trial.user_attrs.get(STOPPED_EPOCH_KEY)
        if stopped_epoch is not None:
            old_num_epochs = pipeline_config['training_kwargs']['num_epochs']
            metadata['_stopper_comment'] = (
                f'While the original config had {old_num_epochs},'
                f' early stopping will now switch it to {int(stopped_epoch)}'
            )
            pipeline_config['training_kwargs']['num_epochs'] = int(stopped_epoch)
        return dict(metadata=metadata, pipeline=pipeline_config)

    def save_to_directory(self, directory: str, **kwargs) -> None:
        """Dump the results of a study to the given directory."""
        os.makedirs(directory, exist_ok=True)

        # Output study information
        with open(os.path.join(directory, 'study.json'), 'w') as file:
            json.dump(self.study.user_attrs, file, indent=2, sort_keys=True)

        # Output all trials
        df = self.study.trials_dataframe()
        df.to_csv(os.path.join(directory, 'trials.tsv'), sep='\t', index=False)

        best_pipeline_directory = os.path.join(directory, 'best_pipeline')
        os.makedirs(best_pipeline_directory, exist_ok=True)
        # Output best trial as pipeline configuration file
        with open(os.path.join(best_pipeline_directory, 'pipeline_config.json'), 'w') as file:
            json.dump(self._get_best_study_config(), file, indent=2, sort_keys=True)

    def save_to_ftp(self, directory: str, ftp: ftplib.FTP):
        """Save the results to the directory in an FTP server.

        :param directory: The directory in the FTP server to save to
        :param ftp: A connection to the FTP server
        """
        ensure_ftp_directory(ftp=ftp, directory=directory)

        study_path = os.path.join(directory, 'study.json')
        ftp.storbinary(f'STOR {study_path}', get_json_bytes_io(self.study.user_attrs))

        trials_path = os.path.join(directory, 'trials.tsv')
        ftp.storbinary(f'STOR {trials_path}', get_df_io(self.study.trials_dataframe()))

        best_pipeline_directory = os.path.join(directory, 'best_pipeline')
        ensure_ftp_directory(ftp=ftp, directory=best_pipeline_directory)

        best_config_path = os.path.join(best_pipeline_directory, 'pipeline_config.json')
        ftp.storbinary(f'STOR {best_config_path}', get_json_bytes_io(self._get_best_study_config()))

    def save_to_s3(self, directory: str, bucket: str, s3=None) -> None:
        """Save all artifacts to the given directory in an S3 Bucket.

        :param directory: The directory in the S3 bucket
        :param bucket: The name of the S3 bucket
        :param s3: A client from :func:`boto3.client`, if already instantiated
        """
        if s3 is None:
            import boto3
            s3 = boto3.client('s3')

        study_path = os.path.join(directory, 'study.json')
        s3.upload_fileobj(get_json_bytes_io(self.study.user_attrs), bucket, study_path)

        trials_path = os.path.join(directory, 'trials.tsv')
        s3.upload_fileobj(get_df_io(self.study.trials_dataframe()), bucket, trials_path)

        best_config_path = os.path.join(directory, 'best_pipeline', 'pipeline_config.json')
        s3.upload_fileobj(get_json_bytes_io(self._get_best_study_config()), bucket, best_config_path)

    def replicate_best_pipeline(
        self,
        *,
        directory: str,
        replicates: int,
        move_to_cpu: bool = False,
        save_replicates: bool = True,
    ) -> None:
        """Run the pipeline on the best configuration, but this time on the "test" set instead of "evaluation" set.

        :param directory: Output directory
        :param replicates: The number of times to retrain the model
        :param move_to_cpu: Should the model be moved back to the CPU? Only relevant if training on GPU.
        :param save_replicates: Should the artifacts of the replicates be saved?
        """
        config = self._get_best_study_config()

        if 'use_testing_data' in config:
            raise ValueError('use_testing_data not be set in the configuration at at all!')

        replicate_pipeline_from_config(
            config=config,
            directory=directory,
            replicates=replicates,
            use_testing_data=True,
            move_to_cpu=move_to_cpu,
            save_replicates=save_replicates,
        )


def hpo_pipeline_from_path(path: str, **kwargs) -> HpoPipelineResult:
    """Run a HPO study from the configuration at the given path."""
    with open(path) as file:
        config = json.load(file)
    return hpo_pipeline_from_config(config, **kwargs)


def hpo_pipeline_from_config(config: Mapping[str, Any], **kwargs) -> HpoPipelineResult:
    """Run the HPO pipeline using a properly formatted configuration dictionary."""
    return hpo_pipeline(
        **config['pipeline'],
        **config['optuna'],
        **kwargs,
    )


def hpo_pipeline(
    *,
    # 1. Dataset
    dataset: Union[None, str, DataSet, Type[DataSet]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training: Union[None, str, TriplesFactory] = None,
    testing: Union[None, str, TriplesFactory] = None,
    validation: Union[None, str, TriplesFactory] = None,
    evaluation_entity_whitelist: Optional[Collection[str]] = None,
    evaluation_relation_whitelist: Optional[Collection[str]] = None,
    # 2. Model
    model: Union[str, Type[Model]],
    model_kwargs: Optional[Mapping[str, Any]] = None,
    model_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    # 3. Loss
    loss: Union[None, str, Type[Loss]] = None,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    loss_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    # 4. Regularizer
    regularizer: Union[None, str, Type[Regularizer]] = None,
    regularizer_kwargs: Optional[Mapping[str, Any]] = None,
    regularizer_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    # 5. Optimizer
    optimizer: Union[None, str, Type[Optimizer]] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    optimizer_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    # 6. Training Loop
    training_loop: Union[None, str, Type[TrainingLoop]] = None,
    negative_sampler: Union[None, str, Type[NegativeSampler]] = None,
    negative_sampler_kwargs: Optional[Mapping[str, Any]] = None,
    negative_sampler_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    # 7. Training
    training_kwargs: Optional[Mapping[str, Any]] = None,
    training_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    stopper: Union[None, str, Type[Stopper]] = None,
    stopper_kwargs: Optional[Mapping[str, Any]] = None,
    # 8. Evaluation
    evaluator: Union[None, str, Type[Evaluator]] = None,
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    metric: Optional[str] = None,
    # 9. Tracking
    result_tracker: Union[None, str, Type[ResultTracker]] = None,
    result_tracker_kwargs: Optional[Mapping[str, Any]] = None,
    # 6. Misc
    device: Union[None, str, torch.device] = None,
    #  Optuna Study Settings
    storage: Union[None, str, BaseStorage] = None,
    sampler: Union[None, str, Type[BaseSampler]] = None,
    sampler_kwargs: Optional[Mapping[str, Any]] = None,
    pruner: Union[None, str, Type[BasePruner]] = None,
    pruner_kwargs: Optional[Mapping[str, Any]] = None,
    study_name: Optional[str] = None,
    direction: Optional[str] = None,
    load_if_exists: bool = False,
    # Optuna Optimization Settings
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    n_jobs: Optional[int] = None,
    save_model_directory: Optional[str] = None,
) -> HpoPipelineResult:
    """Train a model on the given dataset.

    :param dataset:
        The name of the dataset (a key from :data:`pykeen.datasets.datasets`) or the :class:`pykeen.datasets.DataSet`
        instance. Alternatively, the ``training_triples_factory`` and ``testing_triples_factory`` can be specified.
    :param dataset_kwargs:
        The keyword arguments passed to the dataset upon instantiation
    :param training:
        A triples factory with training instances or path to the training file if a a dataset was not specified
    :param testing:
        A triples factory with test instances or path to the test file if a dataset was not specified
    :param validation:
        A triples factory with validation instances or path to the validation file if a dataset was not specified
    :param evaluation_entity_whitelist:
        Optional restriction of evaluation to triples containing *only* these entities. Useful if the downstream task
        is only interested in certain entities, but the relational patterns with other entities improve the entity
        embedding quality. Passed to :func:`pykeen.pipeline.pipeline`.
    :param evaluation_relation_whitelist:
        Optional restriction of evaluation to triples containing *only* these relations. Useful if the downstream task
        is only interested in certain relation, but the relational patterns with other relations improve the entity
        embedding quality. Passed to :func:`pykeen.pipeline.pipeline`.

    :param model:
        The name of the model or the model class to pass to :func:`pykeen.pipeline.pipeline`
    :param model_kwargs:
        Keyword arguments to pass to the model class on instantiation
    :param model_kwargs_ranges:
        Strategies for optimizing the models' hyper-parameters to override
        the defaults

    :param loss:
        The name of the loss or the loss class to pass to :func:`pykeen.pipeline.pipeline`
    :param loss_kwargs:
        Keyword arguments to pass to the loss on instantiation
    :param loss_kwargs_ranges:
        Strategies for optimizing the losses' hyper-parameters to override
        the defaults

    :param regularizer:
        The name of the regularizer or the regularizer class to pass to :func:`pykeen.pipeline.pipeline`
    :param regularizer_kwargs:
        Keyword arguments to pass to the regularizer on instantiation
    :param regularizer_kwargs_ranges:
        Strategies for optimizing the regularizers' hyper-parameters to override
        the defaults

    :param optimizer:
        The name of the optimizer or the optimizer class. Defaults to :class:`torch.optim.Adagrad`.
    :param optimizer_kwargs:
        Keyword arguments to pass to the optimizer on instantiation
    :param optimizer_kwargs_ranges:
        Strategies for optimizing the optimizers' hyper-parameters to override
        the defaults

    :param training_loop:
        The name of the training approach (``'slcwa'`` or ``'lcwa'``) or the training loop class
        to pass to :func:`pykeen.pipeline.pipeline`
    :param negative_sampler:
        The name of the negative sampler (``'basic'`` or ``'bernoulli'``) or the negative sampler class
        to pass to :func:`pykeen.pipeline.pipeline`. Only allowed when training with sLCWA.
    :param negative_sampler_kwargs:
        Keyword arguments to pass to the negative sampler class on instantiation
    :param negative_sampler_kwargs_ranges:
        Strategies for optimizing the negative samplers' hyper-parameters to override
        the defaults

    :param training_kwargs:
        Keyword arguments to pass to the training loop's train function on call
    :param training_kwargs_ranges:
        Strategies for optimizing the training loops' hyper-parameters to override
        the defaults. Can not specify ranges for batch size if early stopping is enabled.

    :param stopper:
        What kind of stopping to use. Default to no stopping, can be set to 'early'.
    :param stopper_kwargs:
        Keyword arguments to pass to the stopper upon instantiation.

    :param evaluator:
        The name of the evaluator or an evaluator class. Defaults to :class:`pykeen.evaluation.RankBasedEvaluator`.
    :param evaluator_kwargs:
        Keyword arguments to pass to the evaluator on instantiation
    :param evaluation_kwargs:
        Keyword arguments to pass to the evaluator's evaluate function on call

    :param result_tracker:
        The ResultsTracker class or name
    :param result_tracker_kwargs:
        The keyword arguments passed to the results tracker on instantiation

    :param metric:
        The metric to optimize over. Defaults to ``adjusted_mean_rank``.
    :param direction:
        The direction of optimization. Because the default metric is ``adjusted_mean_rank``,
        the default direction is ``minimize``.

    :param n_jobs: The number of parallel jobs. If this argument is set to :obj:`-1`, the number is
                set to CPU counts. If none, defaults to 1.

    .. note::

        The remaining parameters are passed to :func:`optuna.study.create_study`
        or :meth:`optuna.study.Study.optimize`.
    """
    sampler_cls = get_sampler_cls(sampler)
    pruner_cls = get_pruner_cls(pruner)

    if direction is None:
        direction = 'minimize'

    study = create_study(
        storage=storage,
        sampler=sampler_cls(**(sampler_kwargs or {})),
        pruner=pruner_cls(**(pruner_kwargs or {})),
        study_name=study_name,
        direction=direction,
        load_if_exists=load_if_exists,
    )

    # 0. Metadata/Provenance
    study.set_user_attr('pykeen_version', get_version())
    study.set_user_attr('pykeen_git_hash', get_git_hash())
    # 1. Dataset
    study.set_user_attr('dataset', _get_dataset_name(
        dataset=dataset,
        dataset_kwargs=dataset_kwargs,
        training=training,
        testing=testing,
        validation=validation,
    ))

    # 2. Model
    model: Type[Model] = get_model_cls(model)
    study.set_user_attr('model', normalize_string(model.__name__))
    logger.info(f'Using model: {model}')
    # 3. Loss
    loss: Type[Loss] = model.loss_default if loss is None else get_loss_cls(loss)
    study.set_user_attr('loss', normalize_string(loss.__name__, suffix=_LOSS_SUFFIX))
    logger.info(f'Using loss: {loss}')
    # 4. Regularizer
    regularizer: Type[Regularizer] = (
        model.regularizer_default
        if regularizer is None else
        get_regularizer_cls(regularizer)
    )
    study.set_user_attr('regularizer', regularizer.get_normalized_name())
    logger.info(f'Using regularizer: {regularizer}')
    # 5. Optimizer
    optimizer: Type[Optimizer] = get_optimizer_cls(optimizer)
    study.set_user_attr('optimizer', normalize_string(optimizer.__name__))
    logger.info(f'Using optimizer: {optimizer}')
    # 6. Training Loop
    training_loop: Type[TrainingLoop] = get_training_loop_cls(training_loop)
    study.set_user_attr('training_loop', training_loop.get_normalized_name())
    logger.info(f'Using training loop: {training_loop}')
    if training_loop is SLCWATrainingLoop:
        negative_sampler: Optional[Type[NegativeSampler]] = get_negative_sampler_cls(negative_sampler)
        study.set_user_attr('negative_sampler', negative_sampler.get_normalized_name())
        logger.info(f'Using negative sampler: {negative_sampler}')
    else:
        negative_sampler: Optional[Type[NegativeSampler]] = None
    # 7. Training
    stopper: Type[Stopper] = get_stopper_cls(stopper)

    if stopper is EarlyStopper and training_kwargs_ranges and 'epochs' in training_kwargs_ranges:
        raise ValueError('can not use early stopping while optimizing epochs')

    # 8. Evaluation
    evaluator: Type[Evaluator] = get_evaluator_cls(evaluator)
    study.set_user_attr('evaluator', evaluator.get_normalized_name())
    logger.info(f'Using evaluator: {evaluator}')
    if metric is None:
        metric = 'adjusted_mean_rank'
    study.set_user_attr('metric', metric)
    logger.info(f'Attempting to {direction} {metric}')

    # 9. Tracking
    result_tracker: Type[ResultTracker] = get_result_tracker_cls(result_tracker)

    objective = Objective(
        # 1. Dataset
        dataset=dataset,
        dataset_kwargs=dataset_kwargs,
        training=training,
        testing=testing,
        validation=validation,
        evaluation_entity_whitelist=evaluation_entity_whitelist,
        evaluation_relation_whitelist=evaluation_relation_whitelist,
        # 2. Model
        model=model,
        model_kwargs=model_kwargs,
        model_kwargs_ranges=model_kwargs_ranges,
        # 3. Loss
        loss=loss,
        loss_kwargs=loss_kwargs,
        loss_kwargs_ranges=loss_kwargs_ranges,
        # 4. Regularizer
        regularizer=regularizer,
        regularizer_kwargs=regularizer_kwargs,
        regularizer_kwargs_ranges=regularizer_kwargs_ranges,
        # 5. Optimizer
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        optimizer_kwargs_ranges=optimizer_kwargs_ranges,
        # 6. Training Loop
        training_loop=training_loop,
        negative_sampler=negative_sampler,
        negative_sampler_kwargs=negative_sampler_kwargs,
        negative_sampler_kwargs_ranges=negative_sampler_kwargs_ranges,
        # 7. Training
        training_kwargs=training_kwargs,
        training_kwargs_ranges=training_kwargs_ranges,
        stopper=stopper,
        stopper_kwargs=stopper_kwargs,
        # 8. Evaluation
        evaluator=evaluator,
        evaluator_kwargs=evaluator_kwargs,
        evaluation_kwargs=evaluation_kwargs,
        # 9. Tracker
        result_tracker=result_tracker,
        result_tracker_kwargs=result_tracker_kwargs,
        # Optuna Misc.
        metric=metric,
        save_model_directory=save_model_directory,
        # Pipeline Misc.
        device=device,
    )

    # Invoke optimization of the objective function.
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs or 1,
    )

    return HpoPipelineResult(
        study=study,
        objective=objective,
    )


def _get_kwargs(
    trial: Trial,
    prefix: str,
    *,
    default_kwargs_ranges: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    kwargs_ranges: Optional[Mapping[str, Any]] = None,
):
    _kwargs_ranges = dict(default_kwargs_ranges)
    if kwargs_ranges is not None:
        _kwargs_ranges.update(kwargs_ranges)
    return suggest_kwargs(
        trial=trial,
        prefix=prefix,
        kwargs_ranges=_kwargs_ranges,
        kwargs=kwargs,
    )


def suggest_kwargs(
    trial: Trial,
    prefix: str,
    kwargs_ranges: Mapping[str, Any],
    kwargs: Optional[Mapping[str, Any]] = None,
):
    _kwargs = {}
    if kwargs:
        _kwargs.update(kwargs)

    for name, info in kwargs_ranges.items():
        if name in _kwargs:
            continue  # has been set by default, won't be suggested

        prefixed_name = f'{prefix}.{name}'
        dtype, low, high = info['type'], info.get('low'), info.get('high')
        if dtype in {int, 'int'}:
            q, scale = info.get('q'), info.get('scale')
            if scale == 'power_two':
                _kwargs[name] = suggest_discrete_power_two_int(
                    trial=trial,
                    name=prefixed_name,
                    low=low,
                    high=high,
                )
            elif q is not None:
                _kwargs[name] = suggest_discrete_uniform_int(
                    trial=trial,
                    name=prefixed_name,
                    low=low,
                    high=high,
                    q=q,
                )
            else:
                _kwargs[name] = trial.suggest_int(name=prefixed_name, low=low, high=high)

        elif dtype in {float, 'float'}:
            if info.get('scale') == 'log':
                _kwargs[name] = trial.suggest_loguniform(name=prefixed_name, low=low, high=high)
            else:
                _kwargs[name] = trial.suggest_uniform(name=prefixed_name, low=low, high=high)
        elif dtype == 'categorical':
            choices = info['choices']
            _kwargs[name] = trial.suggest_categorical(name=prefixed_name, choices=choices)
        elif dtype in {bool, 'bool'}:
            _kwargs[name] = trial.suggest_categorical(name=prefixed_name, choices=[True, False])
        else:
            logger.warning(f'Unhandled data type ({dtype}) for parameter {name}')

    return _kwargs


def suggest_discrete_uniform_int(trial: Trial, name, low, high, q) -> int:
    """Suggest an integer in the given range [low, high] inclusive with step size q."""
    if (high - low) % q:
        logger.warning(f'bad range given: range({low}, {high}, {q}) - not divisible by q')
    choices = list(range(low, high + 1, q))
    return trial.suggest_categorical(name=name, choices=choices)


def suggest_discrete_power_two_int(trial: Trial, name, low, high) -> int:
    """Suggest an integer in the given range [2^low, 2^high]."""
    if high <= low:
        raise Exception(f"Upper bound {high} is not greater than lower bound {low}.")
    choices = [2 ** i for i in range(low, high + 1)]
    return trial.suggest_categorical(name=name, choices=choices)


def _get_dataset_name(
    *,
    dataset: Union[None, str, DataSet, Type[DataSet]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training: Union[None, str, TriplesFactory] = None,
    testing: Union[None, str, TriplesFactory] = None,
    validation: Union[None, str, TriplesFactory] = None,
) -> str:
    """Make a useful name for the dataset for storage in HPO."""
    if (
        (isinstance(dataset, str) and has_dataset(dataset))
        or isinstance(dataset, DataSet)
        or (isinstance(dataset, type) and issubclass(dataset, DataSet))
    ):
        return get_dataset(dataset=dataset).get_normalized_name()

    # TODO make more informative
    return '<user defined>'
