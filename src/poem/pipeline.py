# -*- coding: utf-8 -*-

"""A high level pipeline for training and evaluating a model."""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Type, Union

from torch.optim import Adam, SGD
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from poem.datasets import DataSet, datasets
from poem.evaluation import Evaluator, MetricResults, RankBasedEvaluator
from poem.instance_creation_factories import TriplesFactory
from poem.models import BaseModule
from poem.negative_sampling import BasicNegativeSampler, BernoulliNegativeSampler, NegativeSampler
from poem.training import CWATrainingLoop, OWATrainingLoop, TrainingLoop

__all__ = [
    'PipelineResult',
    'pipeline',
]


def _normalize_string(s):
    return s.lower().replace('-', '').replace('_', '')


def _make_class_lookup(classes: List[type]):
    return {
        _normalize_string(cls.__name__): cls
        for cls in classes
    }


_models = _make_class_lookup(BaseModule.__subclasses__())

# TODO add more optimizers, or get with Optimizer.__subclasses__()?
_optimizer_list = [
    Adam,
    SGD,
    AdamW,
    Adagrad,
    Adadelta,
    Adamax
]
_optimizers = _make_class_lookup(_optimizer_list)

_evaluator_list = [
    RankBasedEvaluator,
]
_evaluators = _make_class_lookup(_evaluator_list)

_training_loops = {
    _normalize_string('owa'): OWATrainingLoop,
    _normalize_string('cwa'): CWATrainingLoop,
}

_negative_samplers = {
    'basic': BasicNegativeSampler,
    'bernoulli': BernoulliNegativeSampler,
}


@dataclass
class PipelineResult:
    """A dataclass containing the results of running :func:`poem.pipeline.pipeline`."""

    #: The model trained by the pipeline
    model: BaseModule

    #: The training loop used by the pipeline
    training_loop: TrainingLoop

    #: The results evaluated by the pipeline
    metric_results: MetricResults


def _not_str_or_type(x):
    return not isinstance(x, (str, type))


def pipeline(  # noqa: C901
        model: Union[str, Type[BaseModule]],
        *,
        optimizer: Union[str, Type[Optimizer]] = Adagrad,
        training_loop: Union[str, Type[TrainingLoop]] = OWATrainingLoop,
        dataset: Union[None, str, DataSet] = None,
        training_triples_factory: Optional[TriplesFactory] = None,
        testing_triples_factory: Optional[TriplesFactory] = None,
        negative_sampler: Union[None, str, Type[NegativeSampler]] = None,
        evaluator: Union[str, Type[Evaluator]] = RankBasedEvaluator,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        training_kwargs: Optional[Mapping[str, Any]] = None,
        evaluator_kwargs: Optional[Mapping[str, Any]] = None,
        evaluation_kwargs: Optional[Mapping[str, Any]] = None,
) -> PipelineResult:
    """Train and evaluate a model.

    :param model: The name of the model or the model class
    :param optimizer: The name of the optimizer or the optimizer class
    :param dataset: The name of the dataset (a key from :data:`poem.datasets.datasets`)
     or the :class:`poem.datasets.DataSet` instance. Alternatively, the ``training_triples_factory`` and
     ``testing_triples_factory`` can be specified.
    :param training_triples_factory: A triples factory with training instances if a
     a dataset was not specified
    :param testing_triples_factory: A triples factory with training instances if a
     dataset was not specified
    :param training_loop: The name of the training loop's assumpiton ('owa' or 'cwa')
     or the training loop class.
    :param negative_sampler: The name of the negative sampler ('basic' or 'bernoulli')
     or the negative sampler class
    :param evaluator: The name of the evaluator or an evaluator class. Defaults to
     rank based evaluator
    :param model_kwargs: Keyword arguments to pass to the model class on instantiation
    :param optimizer_kwargs: Keyword arguments to pass to the optimizer on instantiation
    :param training_kwargs: Keyword arguments to pass to the training loop's train
     function on call
    :param evaluator_kwargs: Keyword arguments to pass to the evaluator on instantiation
    :param evaluation_kwargs: Keyword arguments to pass to the evaluator's evaluate
     function on call

    Train and evaluate TransE on the Nations dataset with the default training
    assumption (``'OWA'``), the default optimizer (``'Adagrad'``), and the default
    evaluator (``'RankBasedEvaluator'``).

    >>> from poem.pipeline import pipeline
    >>> pipeline_result = pipeline(
    ...     model='TransE',
    ...     dataset='nations',
    ... )

    Arguments for the model can be given as a dictionary using
    ``model_kwargs``. There are several other options for passing kwargs in to
    the other parameters used by :func:`pipeline`.

    >>> from poem.pipeline import pipeline
    >>> pipeline_result = pipeline(
    ...     model='TransE',
    ...     dataset='nations',
    ...     model_kwargs=dict(
    ...         scoring_fct_norm=2,
    ...     ),
    ... )

    Train and evaluate TransE on the Nations dataset by explicitly setting all
    settings.

    >>> from poem.pipeline import pipeline
    >>> pipeline_result = pipeline(
    ...     model='TransE',
    ...     dataset='nations',
    ...     optimizer='Adam',
    ...     training_loop='OWA',
    ...     evaluator='RankBasedEvaluator',
    ... )

    Pre-packaged datasets using :class:`poem.datasets.DataSet` can be used
    instead of a name.

    >>> from poem.datasets import nations
    >>> from poem.pipeline import pipeline
    >>> pipeline_result = pipeline(
    ...     model='TransE',
    ...     dataset=nations,
    ... )

    The triples factories for training and testing can be set explicitly.

    >>> from poem.datasets import NationsTestingTriplesFactory
    >>> from poem.datasets import NationsTrainingTriplesFactory
    >>> from poem.pipeline import pipeline
    >>> pipeline_result = pipeline(
    ...     model='TransE',
    ...     training_triples_factory=NationsTrainingTriplesFactory(),
    ...     testing_triples_factory=NationsTestingTriplesFactory(),
    ... )
    """
    if dataset is not None:
        if testing_triples_factory is not None and training_triples_factory is not None:
            raise ValueError('Can not specify both dataset and training_triples_factory/testing_triples_factory.')
        elif training_triples_factory is not None:
            raise ValueError('Can not specify training_triples_factory after specifying dataset.')
        elif testing_triples_factory is not None:
            raise ValueError('Can not specify testing_triples_factory after specifying dataset.')

        if isinstance(dataset, str):
            try:
                dataset = datasets[dataset]
            except KeyError:
                raise ValueError(f'Invalid dataset name: {dataset}')
        dataset.load()
        training_triples_factory = dataset.training
        testing_triples_factory = dataset.testing
    elif testing_triples_factory is None or training_triples_factory is None:
        raise ValueError('Must specify either dataset or both training_triples_factory and testing_triples_factory.')

    if _not_str_or_type(model):
        raise TypeError(f'Invalid model type: {type(model)}')
    elif isinstance(model, str):
        try:
            model = _models[_normalize_string(model)]
        except KeyError:
            raise ValueError(f'Invalid model name: {model}')
    elif not issubclass(model, BaseModule):
        raise TypeError(f'Not subclass of BaseModule: {model}')

    model_instance: BaseModule = model(
        triples_factory=training_triples_factory,
        **(model_kwargs or {}),
    )

    if _not_str_or_type(optimizer):
        raise TypeError(f'Invalid optimizer type: {type(optimizer)} - {optimizer}')
    elif isinstance(optimizer, str):
        try:
            optimizer = _optimizers[_normalize_string(optimizer)]
        except KeyError:
            raise ValueError(f'Invalid optimizer name: {optimizer}')
    elif not issubclass(optimizer, Optimizer):
        raise TypeError(f'Not subclass of Optimizer: {optimizer}')

    optimizer_instance: Optimizer = optimizer(
        params=model_instance.get_grad_params(),
        **(optimizer_kwargs or {}),
    )

    # Pick a training assumption (OWA or CWA)
    if _not_str_or_type(training_loop):
        raise TypeError(f'Invalid training loop type: {type(training_loop)} - {training_loop}')
    elif isinstance(training_loop, str):
        try:
            training_loop = _training_loops[_normalize_string(training_loop)]
        except KeyError:
            raise ValueError(f'Invalid training loop name: {training_loop}')
    elif not issubclass(training_loop, TrainingLoop):
        raise TypeError(f'Not subclass of TrainingLoop: {training_loop}')

    if negative_sampler is None:
        training_loop_instance: TrainingLoop = training_loop(
            model=model_instance,
            optimizer=optimizer_instance,
        )
    else:
        if training_loop is not OWATrainingLoop:
            raise ValueError('Can not specify negative sampler with CWA')
        elif _not_str_or_type(negative_sampler):
            raise TypeError(f'Invalid training negative sampler type: {type(negative_sampler)} - {negative_sampler}')
        elif isinstance(negative_sampler, str):
            try:
                negative_sampler = _negative_samplers[_normalize_string(negative_sampler)]
            except KeyError:
                raise ValueError(f'Invalid negative sampler name: {negative_sampler}')
        elif not issubclass(negative_sampler, NegativeSampler):
            raise TypeError(f'Not subclass of NegativeSampler: {negative_sampler}')

        training_loop_instance: TrainingLoop = training_loop(
            model=model_instance,
            optimizer=optimizer_instance,
            negative_sampler_cls=negative_sampler,
        )

    if training_kwargs is None:
        training_kwargs = {}
    training_kwargs.setdefault('num_epochs', 5)
    training_kwargs.setdefault('batch_size', 256)

    # Train like Cristiano Ronaldo
    training_loop_instance.train(**training_kwargs)

    # Pick an evaluator
    if _not_str_or_type(evaluator):
        raise TypeError(f'Invalid evaluator type: {type(evaluator)} - {evaluator}')
    elif isinstance(evaluator, str):
        try:
            evaluator = _evaluators[_normalize_string(evaluator)]
        except KeyError:
            raise ValueError(f'Invalid evaluator name: {evaluator}')
    elif not issubclass(evaluator, Evaluator):
        raise TypeError(f'Not subclass of Evaluator: {evaluator}')

    evaluator_instance: Evaluator = evaluator(
        model=model_instance,
        **(evaluator_kwargs or {}),
    )

    # Evaluate
    metric_results = evaluator_instance.evaluate(
        testing_triples_factory.mapped_triples,
        **(evaluation_kwargs or {}),
    )

    return PipelineResult(
        model=model_instance,
        training_loop=training_loop_instance,
        metric_results=metric_results,
    )
