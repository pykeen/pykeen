# -*- coding: utf-8 -*-

"""Hyper-parameter optimiziation in PyKEEN."""

import dataclasses
import ftplib
import json
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Collection, Dict, Mapping, Optional, Type, Union

import torch
from optuna import Study, Trial, create_study
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage

from .pruners import pruner_resolver
from .samplers import sampler_resolver
from ..constants import USER_DEFINED_CODE
from ..datasets import get_dataset, has_dataset
from ..datasets.base import Dataset
from ..evaluation import Evaluator, evaluator_resolver
from ..evaluation.rank_based_evaluator import ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX
from ..losses import Loss, loss_resolver
from ..models import Model, model_resolver
from ..optimizers import Optimizer, optimizer_resolver, optimizers_hpo_defaults
from ..pipeline import pipeline, replicate_pipeline_from_config
from ..regularizers import Regularizer, regularizer_resolver
from ..sampling import NegativeSampler, negative_sampler_resolver
from ..stoppers import EarlyStopper, Stopper, stopper_resolver
from ..trackers import ResultTracker, tracker_resolver
from ..training import SLCWATrainingLoop, TrainingLoop, training_loop_resolver
from ..triples import CoreTriplesFactory
from ..typing import Hint, HintType
from ..utils import Result, ensure_ftp_directory, fix_dataclass_init_docs, get_df_io, get_json_bytes_io
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

    dataset: Union[None, str, Dataset, Type[Dataset]]  # 1.
    model: Type[Model]  # 2.
    loss: Type[Loss]  # 3.
    optimizer: Type[Optimizer]  # 5.
    training_loop: Type[TrainingLoop]  # 6.
    stopper: Type[Stopper]  # 7.
    evaluator: Type[Evaluator]  # 8.
    result_tracker: Type[ResultTracker]  # 9.
    metric: str

    # 1. Dataset
    dataset_kwargs: Optional[Mapping[str, Any]] = None
    training: Hint[CoreTriplesFactory] = None
    testing: Hint[CoreTriplesFactory] = None
    validation: Hint[CoreTriplesFactory] = None
    evaluation_entity_whitelist: Optional[Collection[str]] = None
    evaluation_relation_whitelist: Optional[Collection[str]] = None
    # 2. Model
    model_kwargs: Optional[Mapping[str, Any]] = None
    model_kwargs_ranges: Optional[Mapping[str, Any]] = None
    # 3. Loss
    loss_kwargs: Optional[Mapping[str, Any]] = None
    loss_kwargs_ranges: Optional[Mapping[str, Any]] = None
    # 4. Regularizer
    regularizer: Optional[Type[Regularizer]] = None
    regularizer_kwargs: Optional[Mapping[str, Any]] = None
    regularizer_kwargs_ranges: Optional[Mapping[str, Any]] = None
    # 5. Optimizer
    optimizer_kwargs: Optional[Mapping[str, Any]] = None
    optimizer_kwargs_ranges: Optional[Mapping[str, Any]] = None
    # 6. Training Loop
    training_loop_kwargs: Optional[Mapping[str, Any]] = None
    negative_sampler: Optional[Type[NegativeSampler]] = None
    negative_sampler_kwargs: Optional[Mapping[str, Any]] = None
    negative_sampler_kwargs_ranges: Optional[Mapping[str, Any]] = None
    # 7. Training
    training_kwargs: Optional[Mapping[str, Any]] = None
    training_kwargs_ranges: Optional[Mapping[str, Any]] = None
    stopper_kwargs: Optional[Mapping[str, Any]] = None
    # 8. Evaluation
    evaluator_kwargs: Optional[Mapping[str, Any]] = None
    evaluation_kwargs: Optional[Mapping[str, Any]] = None
    filter_validation_when_testing: bool = True
    # 9. Trackers
    result_tracker_kwargs: Optional[Mapping[str, Any]] = None
    # Misc.
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
        _regularizer_kwargs: Optional[Mapping[str, Any]]
        if self.regularizer is None:
            _regularizer_kwargs = {}
        else:
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

        _negative_sampler_kwargs: Mapping[str, Any]
        if self.training_loop is not SLCWATrainingLoop:
            _negative_sampler_kwargs = {}
        else:
            _negative_sampler_kwargs = _get_kwargs(
                trial=trial,
                prefix='negative_sampler',
                default_kwargs_ranges={} if self.negative_sampler is None else self.negative_sampler.hpo_default,
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
                training_loop_kwargs=self.training_loop_kwargs,
                training_kwargs=_training_kwargs,
                stopper=self.stopper,
                stopper_kwargs=_stopper_kwargs,
                # 8. Evaluation
                evaluator=self.evaluator,
                evaluator_kwargs=self.evaluator_kwargs,
                evaluation_kwargs=self.evaluation_kwargs,
                filter_validation_when_testing=self.filter_validation_when_testing,
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
            field_value = getattr(self.objective, field.name)
            if not field_value:
                continue
            if field.name.endswith('_kwargs'):
                logger.debug(f'saving pre-specified field in pipeline config: {field.name}={field_value}')
                pipeline_config[field.name] = field_value
            elif field.name in {'training', 'testing', 'validation'}:
                pipeline_config[field.name] = field_value if isinstance(field_value, str) else USER_DEFINED_CODE

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

    def save_to_directory(self, directory: Union[str, pathlib.Path], **kwargs) -> None:
        """Dump the results of a study to the given directory."""
        if isinstance(directory, str):
            directory = pathlib.Path(directory).resolve()
        directory.mkdir(exist_ok=True, parents=True)

        # Output study information
        with directory.joinpath('study.json').open('w') as file:
            json.dump(self.study.user_attrs, file, indent=2, sort_keys=True)

        # Output all trials
        df = self.study.trials_dataframe()
        df.to_csv(directory.joinpath('trials.tsv'), sep='\t', index=False)

        best_pipeline_directory = directory.joinpath('best_pipeline')
        best_pipeline_directory.mkdir(exist_ok=True, parents=True)
        # Output best trial as pipeline configuration file
        with best_pipeline_directory.joinpath('pipeline_config.json').open('w') as file:
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
        directory: Union[str, pathlib.Path],
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


def hpo_pipeline_from_path(path: Union[str, pathlib.Path], **kwargs) -> HpoPipelineResult:
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
    dataset: Union[None, str, Dataset, Type[Dataset]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training: Hint[CoreTriplesFactory] = None,
    testing: Hint[CoreTriplesFactory] = None,
    validation: Hint[CoreTriplesFactory] = None,
    evaluation_entity_whitelist: Optional[Collection[str]] = None,
    evaluation_relation_whitelist: Optional[Collection[str]] = None,
    # 2. Model
    model: Union[str, Type[Model]],
    model_kwargs: Optional[Mapping[str, Any]] = None,
    model_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    # 3. Loss
    loss: HintType[Loss] = None,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    loss_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    # 4. Regularizer
    regularizer: HintType[Regularizer] = None,
    regularizer_kwargs: Optional[Mapping[str, Any]] = None,
    regularizer_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    # 5. Optimizer
    optimizer: HintType[Optimizer] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    optimizer_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    # 6. Training Loop
    training_loop: HintType[TrainingLoop] = None,
    training_loop_kwargs: Optional[Mapping[str, Any]] = None,
    negative_sampler: HintType[NegativeSampler] = None,
    negative_sampler_kwargs: Optional[Mapping[str, Any]] = None,
    negative_sampler_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    # 7. Training
    training_kwargs: Optional[Mapping[str, Any]] = None,
    training_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    stopper: HintType[Stopper] = None,
    stopper_kwargs: Optional[Mapping[str, Any]] = None,
    # 8. Evaluation
    evaluator: HintType[Evaluator] = None,
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    metric: Optional[str] = None,
    filter_validation_when_testing: bool = True,
    # 9. Tracking
    result_tracker: HintType[ResultTracker] = None,
    result_tracker_kwargs: Optional[Mapping[str, Any]] = None,
    # 6. Misc
    device: Hint[torch.device] = None,
    #  Optuna Study Settings
    storage: Hint[BaseStorage] = None,
    sampler: HintType[BaseSampler] = None,
    sampler_kwargs: Optional[Mapping[str, Any]] = None,
    pruner: HintType[BasePruner] = None,
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
        The name of the dataset (a key for the :data:`pykeen.datasets.dataset_resolver`) or the
        :class:`pykeen.datasets.Dataset` instance. Alternatively, the training triples factory (``training``), testing
        triples factory (``testing``), and validation triples factory (``validation``; optional) can be specified.
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
    :param filter_validation_when_testing:
        If true, during evaluating on the test dataset, validation triples are added to the set of known positive
        triples, which are filtered out when performing filtered evaluation following the approach described by
        [bordes2013]_. Defaults to true.

    :param result_tracker:
        The ResultsTracker class or name
    :param result_tracker_kwargs:
        The keyword arguments passed to the results tracker on instantiation

    :param metric:
        The metric to optimize over. Defaults to :data:`ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX`.
    :param direction:
        The direction of optimization. Because the default metric is :data:`ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX`,
        the default direction is ``maximize``.

    :param n_jobs: The number of parallel jobs. If this argument is set to :obj:`-1`, the number is
                set to CPU counts. If none, defaults to 1.

    .. note::

        The remaining parameters are passed to :func:`optuna.study.create_study`
        or :meth:`optuna.study.Study.optimize`.
    """
    if direction is None:
        direction = 'maximize'

    study = create_study(
        storage=storage,
        sampler=sampler_resolver.make(sampler, sampler_kwargs),
        pruner=pruner_resolver.make(pruner, pruner_kwargs),
        study_name=study_name,
        direction=direction,
        load_if_exists=load_if_exists,
    )

    # 0. Metadata/Provenance
    study.set_user_attr('pykeen_version', get_version())
    study.set_user_attr('pykeen_git_hash', get_git_hash())
    # 1. Dataset
    _set_study_dataset(
        study=study,
        dataset=dataset,
        training=training,
        testing=testing,
        validation=validation,
    )

    # 2. Model
    model_cls: Type[Model] = model_resolver.lookup(model)
    study.set_user_attr('model', model_resolver.normalize_cls(model_cls))
    logger.info(f'Using model: {model_cls}')
    # 3. Loss
    loss_cls: Type[Loss] = model_cls.loss_default if loss is None else loss_resolver.lookup(loss)
    study.set_user_attr('loss', loss_resolver.normalize_cls(loss_cls))
    logger.info(f'Using loss: {loss_cls}')
    # 4. Regularizer
    regularizer_cls: Optional[Type[Regularizer]]
    if regularizer is not None:
        regularizer_cls = regularizer_resolver.lookup(regularizer)
    elif getattr(model_cls, 'regularizer_default', None):
        regularizer_cls = model_cls.regularizer_default
    else:
        regularizer_cls = None
    if regularizer_cls:
        study.set_user_attr('regularizer', regularizer_cls.get_normalized_name())
        logger.info(f'Using regularizer: {regularizer_cls}')
    # 5. Optimizer
    optimizer_cls: Type[Optimizer] = optimizer_resolver.lookup(optimizer)
    study.set_user_attr('optimizer', optimizer_resolver.normalize_cls(optimizer_cls))
    logger.info(f'Using optimizer: {optimizer_cls}')
    # 6. Training Loop
    training_loop_cls: Type[TrainingLoop] = training_loop_resolver.lookup(training_loop)
    study.set_user_attr('training_loop', training_loop_cls.get_normalized_name())
    logger.info(f'Using training loop: {training_loop_cls}')
    negative_sampler_cls: Optional[Type[NegativeSampler]]
    if training_loop_cls is SLCWATrainingLoop:
        negative_sampler_cls = negative_sampler_resolver.lookup(negative_sampler)
        assert negative_sampler_cls is not None
        study.set_user_attr('negative_sampler', negative_sampler_cls.get_normalized_name())
        logger.info(f'Using negative sampler: {negative_sampler_cls}')
    else:
        negative_sampler_cls = None
    # 7. Training
    stopper_cls: Type[Stopper] = stopper_resolver.lookup(stopper)
    if stopper_cls is EarlyStopper and training_kwargs_ranges and 'epochs' in training_kwargs_ranges:
        raise ValueError('can not use early stopping while optimizing epochs')

    # 8. Evaluation
    evaluator_cls: Type[Evaluator] = evaluator_resolver.lookup(evaluator)
    study.set_user_attr('evaluator', evaluator_cls.get_normalized_name())
    logger.info(f'Using evaluator: {evaluator_cls}')
    if metric is None:
        metric = ADJUSTED_ARITHMETIC_MEAN_RANK_INDEX
    study.set_user_attr('metric', metric)
    logger.info(f'Attempting to {direction} {metric}')
    study.set_user_attr('filter_validation_when_testing', filter_validation_when_testing)
    logger.info('Filter validation triples when testing: %s', filter_validation_when_testing)

    # 9. Tracking
    result_tracker_cls: Type[ResultTracker] = tracker_resolver.lookup(result_tracker)

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
        model=model_cls,
        model_kwargs=model_kwargs,
        model_kwargs_ranges=model_kwargs_ranges,
        # 3. Loss
        loss=loss_cls,
        loss_kwargs=loss_kwargs,
        loss_kwargs_ranges=loss_kwargs_ranges,
        # 4. Regularizer
        regularizer=regularizer_cls,
        regularizer_kwargs=regularizer_kwargs,
        regularizer_kwargs_ranges=regularizer_kwargs_ranges,
        # 5. Optimizer
        optimizer=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        optimizer_kwargs_ranges=optimizer_kwargs_ranges,
        # 6. Training Loop
        training_loop=training_loop_cls,
        training_loop_kwargs=training_loop_kwargs,
        negative_sampler=negative_sampler_cls,
        negative_sampler_kwargs=negative_sampler_kwargs,
        negative_sampler_kwargs_ranges=negative_sampler_kwargs_ranges,
        # 7. Training
        training_kwargs=training_kwargs,
        training_kwargs_ranges=training_kwargs_ranges,
        stopper=stopper_cls,
        stopper_kwargs=stopper_kwargs,
        # 8. Evaluation
        evaluator=evaluator_cls,
        evaluator_kwargs=evaluator_kwargs,
        evaluation_kwargs=evaluation_kwargs,
        filter_validation_when_testing=filter_validation_when_testing,
        # 9. Tracker
        result_tracker=result_tracker_cls,
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
    kwargs: Optional[Mapping[str, Any]] = None,
    kwargs_ranges: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
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
) -> Mapping[str, Any]:
    _kwargs: Dict[str, Any] = {}
    if kwargs:
        _kwargs.update(kwargs)

    for name, info in kwargs_ranges.items():
        if name in _kwargs:
            continue  # has been set by default, won't be suggested

        prefixed_name = f'{prefix}.{name}'

        # TODO: make it even easier to specify categorical strategies just as lists
        # if isinstance(info, (tuple, list, set)):
        #     info = dict(type='categorical', choices=list(info))

        dtype, low, high = info['type'], info.get('low'), info.get('high')
        log = info.get('log') in {True, 'TRUE', 'True', 'true', 't', 'YES', 'Yes', 'yes', 'y'}
        if dtype in {int, 'int'}:
            scale = info.get('scale')
            if scale in {'power_two', 'power'}:
                _kwargs[name] = suggest_discrete_power_int(
                    trial=trial,
                    name=prefixed_name,
                    low=low,
                    high=high,
                    base=info.get('q') or info.get('base') or 2,
                )
            elif scale is None or scale == 'linear':
                # get log from info - could either be a boolean or string
                _kwargs[name] = trial.suggest_int(
                    name=prefixed_name,
                    low=low,
                    high=high,
                    step=info.get('q') or info.get('step') or 1,
                    log=log,
                )
            else:
                logger.warning(f'Unhandled scale {scale} for parameter {name} of data type {dtype}')

        elif dtype in {float, 'float'}:
            _kwargs[name] = trial.suggest_float(
                name=prefixed_name,
                low=low,
                high=high,
                step=info.get('q') or info.get('step'),
                log=log,
            )
        elif dtype == 'categorical':
            choices = info['choices']
            _kwargs[name] = trial.suggest_categorical(name=prefixed_name, choices=choices)
        elif dtype in {bool, 'bool'}:
            _kwargs[name] = trial.suggest_categorical(name=prefixed_name, choices=[True, False])
        else:
            logger.warning(f'Unhandled data type ({dtype}) for parameter {name}')

    return _kwargs


def suggest_discrete_power_int(trial: Trial, name: str, low: int, high: int, base: int = 2) -> int:
    """Suggest an integer in the given range [2^low, 2^high]."""
    if high <= low:
        raise Exception(f"Upper bound {high} is not greater than lower bound {low}.")
    choices = [base ** i for i in range(low, high + 1)]
    return trial.suggest_categorical(name=name, choices=choices)


def _set_study_dataset(
    study: Study,
    *,
    dataset: Union[None, str, Dataset, Type[Dataset]] = None,
    training: Union[None, str, CoreTriplesFactory] = None,
    testing: Union[None, str, CoreTriplesFactory] = None,
    validation: Union[None, str, CoreTriplesFactory] = None,
):
    if dataset is not None:
        if training is not None or testing is not None or validation is not None:
            raise ValueError("Cannot specify dataset and training, testing and validation")
        elif isinstance(dataset, (str, pathlib.Path)):
            if isinstance(dataset, str) and has_dataset(dataset):
                study.set_user_attr('dataset', get_dataset(dataset=dataset).get_normalized_name())
            else:
                # otherwise, dataset refers to a file that should be automatically split
                study.set_user_attr('dataset', str(dataset))
        elif (
            isinstance(dataset, Dataset)
            or (isinstance(dataset, type) and issubclass(dataset, Dataset))
        ):
            # this could be custom data, so don't store anything. However, it's possible to check if this
            # was a pre-registered dataset. If that's the desired functionality, we can uncomment the following:
            # dataset_name = dataset.get_normalized_name()  # this works both on instances and classes
            # if has_dataset(dataset_name):
            #     study.set_user_attr('dataset', dataset_name)
            pass
        else:
            raise TypeError(f'Dataset is invalid type: ({type(dataset)}) {dataset}')
    else:
        if isinstance(training, (str, pathlib.Path)):
            study.set_user_attr('training', str(training))
        if isinstance(testing, (str, pathlib.Path)):
            study.set_user_attr('testing', str(testing))
        if isinstance(validation, (str, pathlib.Path)):
            study.set_user_attr('validation', str(validation))
