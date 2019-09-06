# -*- coding: utf-8 -*-

"""The easiest way to train and evaluate a model is with the :func:`poem.pipeline.pipeline` function.

It provides a high-level entry point into the extensible functionality of
this package. The following example shows how to train and evaluate the
TransE model on the Nations dataset.

>>> from poem.pipeline import pipeline
>>> result = pipeline(
...     model='TransE',
...     data_set='Nations',
... )

The results are returned in a :class:`poem.pipeline.PipelineResult` instance, which has
attributes for the trained model, the training loop, and the evaluation.

In this example, the model was given as a string. A list of available models can be found in
:mod:`poem.models`. Alternatively, the class corresponding to the implementation of the model
could be used as in:

>>> from poem.pipeline import pipeline
>>> from poem.models import TransE
>>> result = pipeline(
...     model=TransE,
...     data_set='Nations',
... )

In this example, the data set was given as a string. A list of available data sets can be found in
:mod:`poem.datasets`. Alternatively, the instance of the :class:`poem.datasets.DataSet` could be
used as in:

>>> from poem.pipeline import pipeline
>>> from poem.models import TransE
>>> from poem.datasets import nations
>>> result = pipeline(
...     model=TransE,
...     data_set=nations,
... )

In each of the previous three examples, the training assumption, optimizer, and evaluation scheme
were omitted. By default, the open world assumption (OWA) is used in training. This can be explicitly
given as a string:

>>> from poem.pipeline import pipeline
>>> result = pipeline(
...     model='TransE',
...     data_set='Nations',
...     training_loop='OWA',
... )

Alternatively, the closed world assumption (CWA) can be given with ``'CWA'``. No additional configuration
is necessary, but it's worth reading up on the differences between these assumptions.

>>> from poem.pipeline import pipeline
>>> result = pipeline(
...     model='TransE',
...     data_set='Nations',
...     training_loop='CWA',
... )

One of these differences is that the OWA relies on *negative sampling*. The type of negative sampling
can be given as in:

>>> from poem.pipeline import pipeline
>>> result = pipeline(
...     model='TransE',
...     data_set='Nations',
...     training_loop='OWA',
...     negative_sampler='basic',
... )

In this example, the negative sampler was given as a string. A list of available negative samplers
can be found in :mod:`poem.sampling`. Alternatively, the class corresponding to the implementation
of the negative sampler could be used as in:

>>> from poem.pipeline import pipeline
>>> from poem.sampling import BasicNegativeSampler
>>> result = pipeline(
...     model='TransE',
...     data_set='Nations',
...     training_loop='OWA',
...     negative_sampler=BasicNegativeSampler,
... )

.. warning ::

   The ``negative_sampler`` keyword argument should not be used if the CWA is being used.
   In general, all other options are available under either assumption.

The type of evaluation perfomed can be specified with the ``evaluator`` keyword. By default,
rank-based evaluation is used. It can be given explictly as in:

>>> from poem.pipeline import pipeline
>>> result = pipeline(
...     model='TransE',
...     data_set='Nations',
...     evaluator='RankBasedEvaluator',
... )

In this example, the evaluator string. A list of available evaluators can be found in
:mod:`poem.evaluation`. Alternatively, the class corresponding to the implementation
of the evaluator could be used as in:

>>> from poem.pipeline import pipeline
>>> from poem.evaluation import RankBasedEvaluator
>>> result = pipeline(
...     model='TransE',
...     data_set='Nations',
...     evaluator=RankBasedEvaluator,
... )

POEM implements early stopping, which can be turned on with the ``early_stopping`` keyword
argument as in:

>>> from poem.pipeline import pipeline
>>> result = pipeline(
...     model='TransE',
...     data_set='Nations',
...     early_stopping=True,
... )

Deeper Configuration
~~~~~~~~~~~~~~~~~~~~
Arguments for the model can be given as a dictionary using
``model_kwargs``. There are several other options for passing kwargs in to
the other parameters used by :func:`poem.pipeline.pipeline`.

>>> from poem.pipeline import pipeline
>>> pipeline_result = pipeline(
...     model='TransE',
...     data_set='Nations',
...     model_kwargs=dict(
...         scoring_fct_norm=2,
...     ),
... )

Because the pipeline takes care of looking up classes and instantiating them,
there are several other parameters to :func:`poem.pipeline.pipeline` that
can be used to specify the parameters during their respective instantiations.

Bring Your Own Data
~~~~~~~~~~~~~~~~~~~
As an alternative to using a pre-packaged dataset, the training and testing can be set
explicitly with instances of :class:`poem.triples.TriplesFactory`. For convenience,
the default data sets are also provided as subclasses of :class:`poem.triples.TriplesFactory`.

.. warning ::

    Make sure they are mapped to the same entities.

>>> from poem.datasets import NationsTestingTriplesFactory
>>> from poem.datasets import NationsTrainingTriplesFactory
>>> from poem.pipeline import pipeline
>>> training = NationsTrainingTriplesFactory()
>>> testing = NationsTestingTriplesFactory(
...     entity_to_id=training.entity_to_id,
...     relation_to_id=training.relation_to_id,
... )
>>> pipeline_result = pipeline(
...     model='TransE',
...     training_triples_factory=training,
...     testing_triples_factory=testing,
... )
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Type, Union

from torch.optim import Adam, SGD
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from .datasets import DataSet, datasets
from .evaluation import Evaluator, MetricResults, RankBasedEvaluator
from .models import BaseModule
from .sampling import NegativeSampler, negative_samplers
from .training import EarlyStopper, OWATrainingLoop, TrainingLoop, training_loops
from .triples import TriplesFactory

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
    Adamax,
]
_optimizers = _make_class_lookup(_optimizer_list)

_evaluator_list = [
    RankBasedEvaluator,
]
_evaluators = _make_class_lookup(_evaluator_list)


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
    optimizer: Union[None, str, Type[Optimizer]] = None,
    training_loop: Union[None, str, Type[TrainingLoop]] = None,
    data_set: Union[None, str, DataSet] = None,
    training_triples_factory: Optional[TriplesFactory] = None,
    testing_triples_factory: Optional[TriplesFactory] = None,
    validation_triples_factory: Optional[TriplesFactory] = None,
    negative_sampler: Union[None, str, Type[NegativeSampler]] = None,
    evaluator: Union[None, str, Type[Evaluator]] = None,
    early_stopping: bool = False,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    training_kwargs: Optional[Mapping[str, Any]] = None,
    early_stopping_kwargs: Optional[Mapping[str, Any]] = None,
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
) -> PipelineResult:
    """Train and evaluate a model.

    :param model: The name of the model or the model class
    :param optimizer: The name of the optimizer or the optimizer class.
     Defaults to :class:`torch.optim.Adagrad`.
    :param data_set: The name of the dataset (a key from :data:`poem.datasets.datasets`)
     or the :class:`poem.datasets.DataSet` instance. Alternatively, the ``training_triples_factory`` and
     ``testing_triples_factory`` can be specified.
    :param training_triples_factory: A triples factory with training instances if a
     a dataset was not specified
    :param testing_triples_factory: A triples factory with training instances if a
     dataset was not specified
    :param validation_triples_factory: A triples factory with validation instances if a
     a dataset was not specified
    :param training_loop: The name of the training loop's assumption (``'owa'`` or ``'cwa'``)
     or the training loop class. Defaults to :class:`poem.training.OWATrainingLoop`.
    :param negative_sampler: The name of the negative sampler (``'basic'`` or ``'bernoulli'``)
     or the negative sampler class. Only allowed when training with OWA. Defaults to
     :class:`poem.sampling.BasicNegativeSampler`.
    :param evaluator: The name of the evaluator or an evaluator class. Defaults to
     :class:`poem.evaluation.RankBasedEvaluator`.
    :param early_stopping: Whether to use early stopping. Defaults to false.
    :param model_kwargs: Keyword arguments to pass to the model class on instantiation
    :param optimizer_kwargs: Keyword arguments to pass to the optimizer on instantiation
    :param training_kwargs: Keyword arguments to pass to the training loop's train
     function on call
    :param early_stopping_kwargs: Keyword arguments to pass to the EarlyStopper upon instantiation.
    :param evaluator_kwargs: Keyword arguments to pass to the evaluator on instantiation
    :param evaluation_kwargs: Keyword arguments to pass to the evaluator's evaluate
     function on call
    """
    if data_set is not None:
        if any(f is not None for f in (training_triples_factory, testing_triples_factory, validation_triples_factory)):
            raise ValueError('Can not specify both dataset and any triples factory.')

        if isinstance(data_set, str):
            try:
                data_set = datasets[data_set]
            except KeyError:
                raise ValueError(f'Invalid dataset name: {data_set}')
        training_triples_factory = data_set.training
        testing_triples_factory = data_set.testing
        validation_triples_factory = data_set.validation
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

    if optimizer is None:
        optimizer = Adagrad
    elif _not_str_or_type(optimizer):
        raise TypeError(f'Invalid optimizer type: {type(optimizer)} - {optimizer}')
    elif isinstance(optimizer, str):
        try:
            optimizer = _optimizers[_normalize_string(optimizer)]
        except KeyError:
            raise ValueError(f'Invalid optimizer name: {optimizer}')
    elif not issubclass(optimizer, Optimizer):
        raise TypeError(f'Not subclass of Optimizer: {optimizer}')

    # Pick a training assumption (OWA or CWA)
    if training_loop is None:
        training_loop = OWATrainingLoop
    elif _not_str_or_type(training_loop):
        raise TypeError(f'Invalid training loop type: {type(training_loop)} - {training_loop}')
    elif isinstance(training_loop, str):
        try:
            training_loop = training_loops[_normalize_string(training_loop)]
        except KeyError:
            raise ValueError(f'Invalid training loop name: {training_loop}')
    elif not issubclass(training_loop, TrainingLoop):
        raise TypeError(f'Not subclass of TrainingLoop: {training_loop}')

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer_instance = optimizer(
        params=model_instance.get_grad_params(),
        **optimizer_kwargs
    )

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
                negative_sampler = negative_samplers[_normalize_string(negative_sampler)]
            except KeyError:
                raise ValueError(f'Invalid negative sampler name: {negative_sampler}')
        elif not issubclass(negative_sampler, NegativeSampler):
            raise TypeError(f'Not subclass of NegativeSampler: {negative_sampler}')

        training_loop_instance: TrainingLoop = training_loop(
            model=model_instance,
            optimizer=optimizer_instance,
            negative_sampler_cls=negative_sampler,
        )

    # Pick an evaluator
    if evaluator is None:
        evaluator = RankBasedEvaluator
    elif _not_str_or_type(evaluator):
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

    # Early stopping
    if early_stopping:
        if early_stopping_kwargs is None:
            early_stopping_kwargs = {}
        if validation_triples_factory is None:
            raise ValueError('Must specify a validation_triples_factory or a dataset for using early stopping.')
        early_stopper = EarlyStopper(
            evaluator=evaluator_instance,
            evaluation_triples_factory=validation_triples_factory,
            **early_stopping_kwargs
        )
    else:
        early_stopper = None

    if training_kwargs is None:
        training_kwargs = {}
    training_kwargs.setdefault('num_epochs', 5)
    training_kwargs.setdefault('batch_size', 256)
    training_kwargs.setdefault('early_stopper', early_stopper)

    # Train like Cristiano Ronaldo
    training_loop_instance.train(**training_kwargs)

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
