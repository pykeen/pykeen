# -*- coding: utf-8 -*-

"""Hyper-parameter optimiziation in POEM."""

import dataclasses
import logging
from typing import Any, Callable, Mapping, Optional, Type, Union

from optuna import Study, Trial, create_study
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage

from .samplers import get_sampler_cls
from ..datasets import DataSet
from ..loss_functions import get_loss_cls, losses_hpo_defaults
from ..models import get_model_cls
from ..models.base import BaseModule
from ..pipeline import pipeline
from ..training import OWATrainingLoop, TrainingLoop, get_training_loop_cls
from ..typing import Loss

__all__ = [
    'make_study',
    'make_objective',
]

logger = logging.getLogger(__name__)

Objective = Callable[[Trial], float]


def make_study(
    model: Union[str, Type[BaseModule]],
    data_set: Union[None, str, DataSet],
    *,
    storage: Union[None, str, BaseStorage] = None,
    sampler: Union[None, str, Type[BaseSampler]] = None,
    sampler_kwargs: Optional[Mapping[str, Any]] = None,
    pruner: Optional[BasePruner] = None,
    study_name: Optional[str] = None,
    direction: str = 'minimize',
    load_if_exists: bool = False,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    model_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    criterion: Union[None, str, Type[Loss]] = None,
    criterion_kwargs: Optional[Mapping[str, Any]] = None,
    criterion_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    training_loop: Union[None, str, Type[TrainingLoop]] = None,
    pipeline_kwargs: Optional[Mapping[str, Any]] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    n_jobs: int = 1,
    metric: Optional[str] = None
) -> Study:
    """Train a model on the given dataset.

    :param model: Either an implemented model from :mod:`poem.models` or a list of them.
    :param data_set: A data set to be passed to :func:`poem.pipeline.pipeline`
    :param model_kwargs: Keyword arguments to be passed to the model (that shouldn't be optimized)
    :param model_kwargs_ranges: Ranges for hyperparameters to override the defaults
    :param pipeline_kwargs: Default keyword arguments to be passed to the :func:`poem.pipeline.pipeline` (that
     shouldn't be optimized)
    :param metric: The metric to optimize over. Defaults to ``adjusted_mean_rank``.

    .. note::

        The remaining parameters are passed to :func:`optuna.study.create_study`
        or :meth:`optuna.study.Study.optimize`.

    All of the following examples are about getting the best model
    when training TransE on the Nations data set. Each gives a bit
    of insight into usage of the :func:`make_study` function.

    Run thirty trials:

    >>> from poem.hpo import make_study
    >>> study = make_study(
    ...    model='TransE',  # can also be the model itself
    ...    data_set='nations',
    ...    n_trials=30,
    ... )
    >>> best_model = study.best_trial.user_attrs['model']

    Run as many trials as possible in 60 seconds:

    >>> from poem.hpo import make_study
    >>> study = make_study(
    ...    model='TransE',
    ...    data_set='nations',
    ...    timeout=60,  # this parameter is measured in seconds
    ... )

    Supply some default hyperparameters for TransE that won't be optimized:

    >>> from poem.hpo import make_study
    >>> study = make_study(
    ...    model='TransE',
    ...    model_kwargs=dict(
    ...        embedding_dim=200,
    ...    ),
    ...    data_set='nations',
    ...    n_trials=30,
    ... )

    Supply ranges for some parameters that are different than the defaults:

    >>> from poem.hpo import make_study
    >>> study = make_study(
    ...    model='TransE',
    ...    model_kwargs_ranges=dict(
    ...        embedding_dim=dict(type=int, low=100, high=200, q=25),  # normally low=50, high=350, q=25
    ...    ),
    ...    data_set='nations',
    ...    n_trials=30,
    ... )

    While each model has its own default criteria, specify (explicitly) the criteria with:

    >>> from poem.hpo import make_study
    >>> study = make_study(
    ...    model='TransE',
    ...    model_kwargs_ranges=dict(
    ...        embedding_dim=dict(type=int, low=100, high=200, q=25),  # normally low=50, high=350, q=25
    ...    ),
    ...    criterion='MarginRankingLoss',
    ...    data_set='nations',
    ...    n_trials=30,
    ... )

    Each criterion has its own default hyperparameter optimization ranges, but new ones can
    be set with:

    >>> from poem.hpo import make_study
    >>> study = make_study(
    ...    model='TransE',
    ...    model_kwargs_ranges=dict(
    ...        embedding_dim=dict(type=int, low=100, high=200, q=25),  # normally low=50, high=350, q=25
    ...    ),
    ...    criterion='MarginRankingLoss',
    ...    criterion_kwargs_ranges=dict(
    ...        margin=dict(type=float, low=1.0, high=2.0),
    ...    ),
    ...    data_set='nations',
    ...    n_trials=30,
    ... )

    By default, :mod:`optuna` uses the Tree-structured Parzen Estimator (TPE)
    estimator (:class:`optuna.samplers.TPESampler`), which is a probabilistic
    approach.

    To emulate most hyperparameter optimizations that have used random
    sampling, use :class:`optuna.samplers.RandomSampler` like in:

    >>> from poem.hpo import make_study
    >>> from optuna.samplers import RandomSampler
    >>> study = make_study(
    ...    model='TransE',
    ...    data_set='nations',
    ...    n_trials=30,
    ...    sampler=RandomSampler,
    ... )

    Alternatively, the strings ``"tpe"`` or ``"random"`` can be used so you
    don't have to import :mod:`optuna` in your script.

    >>> from poem.hpo import make_study
    >>> from optuna.samplers import RandomSampler
    >>> study = make_study(
    ...    model='TransE',
    ...    data_set='nations',
    ...    n_trials=30,
    ...    sampler='random',
    ... )

    While :class:`optuna.samplers.RandomSampler` doesn't (currently) take
    any arguments, the ``sampler_kwargs`` parameter can be used to pass
    arguments by keyword to the instantiation of
    :class:`optuna.samplers.TPESampler` like in:

    >>> from poem.hpo import make_study
    >>> from optuna.samplers import RandomSampler
    >>> study = make_study(
    ...    model='TransE',
    ...    data_set='nations',
    ...    n_trials=30,
    ...    sampler='tpe',
    ...    sampler_kwars=dict(prior_weight=1.1),
    ... )

    """
    sampler_cls = get_sampler_cls(sampler)

    study = create_study(
        storage=storage,
        sampler=sampler_cls(**(sampler_kwargs or {})),
        pruner=pruner,
        study_name=study_name,
        direction=direction,
        load_if_exists=load_if_exists,
    )

    if metric is None:
        metric = 'adjusted_mean_rank'
    study.set_user_attr('metric', metric)

    model: Type[BaseModule] = get_model_cls(model)
    study.set_user_attr('model', model.__name__)

    criterion: Type[Loss] = model.criterion_default if criterion is None else get_loss_cls(criterion)
    study.set_user_attr('criterion', criterion.__name__)

    training_loop: Type[TrainingLoop] = get_training_loop_cls(training_loop)
    study.set_user_attr('assumption', training_loop.__name__[:3])

    study.set_user_attr('data_set', data_set)

    objective = make_objective(
        model=model,
        model_kwargs=model_kwargs,
        model_kwargs_ranges=model_kwargs_ranges,
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        criterion_kwargs_ranges=criterion_kwargs_ranges,
        data_set=data_set,
        training_loop=training_loop,
        metric=metric,
        pipeline_kwargs=pipeline_kwargs,
    )

    # Invoke optimization of the objective function.
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
    )

    return study


def make_objective(
    model: Type[BaseModule],
    criterion: Type[Loss],
    training_loop: Type[TrainingLoop],
    data_set: Union[None, str, DataSet],
    metric: str,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    model_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    criterion_kwargs: Optional[Mapping[str, Any]] = None,
    criterion_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    pipeline_kwargs: Optional[Mapping[str, Any]] = None,
) -> Objective:  # noqa: D202
    """Make an objective function for the given model."""

    def objective(trial: Trial) -> float:
        """Suggest parameters then train the model."""
        _model_kwargs_ranges = model.hpo_default.copy()
        if model_kwargs_ranges is not None:
            _model_kwargs_ranges.update(model_kwargs_ranges)

        _model_kwargs = suggest_kwargs(
            trial=trial,
            kwargs_ranges=_model_kwargs_ranges,
            kwargs=model_kwargs,
        )

        if criterion not in losses_hpo_defaults:
            logging.warning('criterion has no default ranges')
            _criterion_kwargs_ranges = {}
        else:
            _criterion_kwargs_ranges = losses_hpo_defaults[criterion]
        if criterion_kwargs_ranges is not None:
            _criterion_kwargs_ranges.update(criterion_kwargs_ranges)

        _criterion_kwargs = suggest_kwargs(
            trial=trial,
            kwargs_ranges=_criterion_kwargs_ranges,
            kwargs=criterion_kwargs,
        )

        if training_loop is OWATrainingLoop:
            # get negative sampler
            logger.debug('HPO for negative sampling not implemented yet')
            pass

        result = pipeline(
            model=model,
            model_kwargs=_model_kwargs,
            criterion=criterion,
            criterion_kwargs=_criterion_kwargs,
            data_set=data_set,
            training_loop=training_loop,
            **(pipeline_kwargs or {}),
        )

        for field in dataclasses.fields(result.metric_results):
            value = getattr(result.metric_results, field.name)
            if field.name == 'hits_at_k':
                for k, hits_at_k in value.items():
                    trial.set_user_attr(f'hits_at_{k}', hits_at_k)
            else:
                trial.set_user_attr(field.name, value)

        if metric.startswith('hits_at_'):
            k = int(metric[len('hits_at_'):])
            return result.metric_results.hits_at_k[k]
        else:
            return getattr(result.metric_results, metric)

    return objective


def suggest_kwargs(
    trial: Trial,
    kwargs_ranges,
    kwargs=None,
):
    _kwargs = {}
    if kwargs:
        _kwargs.update(kwargs)

    for name, info in kwargs_ranges.items():
        if name in _kwargs:
            continue  # has been set by default, won't be suggested

        dtype, low, high = info['type'], info.get('low'), info.get('high')
        if dtype is int:
            q = info.get('q')
            if q is not None:
                _kwargs[name] = int(trial.suggest_discrete_uniform(name=name, low=low, high=high, q=q))
            else:
                _kwargs[name] = trial.suggest_int(name=name, low=low, high=high)
        elif dtype is float:
            if info.get('scale') == 'log':
                _kwargs[name] = trial.suggest_loguniform(name=name, low=low, high=high)
            else:
                _kwargs[name] = trial.suggest_uniform(name=name, low=low, high=high)
        elif dtype == 'categorical':
            choices = info['choices']
            _kwargs[name] = trial.suggest_categorical(name=name, choices=choices)
        elif dtype is bool:
            _kwargs[name] = trial.suggest_categorical(name=name, choices=[True, False])
        else:
            logger.warning(f'Unhandled parameter {name} ({dtype})')

    return _kwargs
