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

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Type, Union

import torch
from torch.optim.optimizer import Optimizer

from .datasets import DataSet, get_data_set
from .evaluation import Evaluator, MetricResults, get_evaluator_cls
from .loss_functions import get_loss_cls
from .models import get_model_cls
from .models.base import BaseModule
from .optimizers import get_optimizer_cls
from .sampling import NegativeSampler, get_negative_sampler_cls
from .training import EarlyStopper, OWATrainingLoop, TrainingLoop, get_training_loop_cls
from .triples import TriplesFactory
from .typing import Loss

__all__ = [
    'PipelineResult',
    'pipeline',
]


@dataclass
class PipelineResult:
    """A dataclass containing the results of running :func:`poem.pipeline.pipeline`."""

    #: The model trained by the pipeline
    model: BaseModule

    #: The training loop used by the pipeline
    training_loop: TrainingLoop

    #: The losses during training
    losses: List[float]

    #: The results evaluated by the pipeline
    metric_results: MetricResults

    #: Any additional metadata as a dictionary
    metadata: Optional[Mapping[str, Any]] = field(default_factory=dict)

    @property
    def title(self) -> Optional[str]:  # noqa:D401
        """The title of the experiment."""
        return self.metadata.get('title')

    def plot_losses(self):
        """Plot the losses per epoch."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        if self.title is not None:
            plt.title(self.title)
        return sns.lineplot(x=range(len(self.losses)), y=self.losses)

    def save_model(self, path) -> None:
        """Save the trained model to the given path using :func:`torch.save`.

        The model contains within it the triples factory that was used for training.
        """
        torch.save(self.model, path)


def pipeline(  # noqa: C901
    model: Union[str, Type[BaseModule]],
    *,
    optimizer: Union[None, str, Type[Optimizer]] = None,
    criterion: Union[None, str, Type[Loss]] = None,
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
    criterion_kwargs: Optional[Mapping[str, Any]] = None,
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
    training_triples_factory, testing_triples_factory, validation_triples_factory = get_data_set(
        data_set=data_set,
        training_triples_factory=training_triples_factory,
        testing_triples_factory=testing_triples_factory,
        validation_triples_factory=validation_triples_factory,
    )

    if model_kwargs is None:
        model_kwargs = {}

    if criterion is not None:
        criterion_cls = get_loss_cls(criterion)
        _criterion = criterion_cls(**(criterion_kwargs or {}))
        model_kwargs.setdefault('criterion', _criterion)

    model = get_model_cls(model)
    model_instance: BaseModule = model(
        triples_factory=training_triples_factory,
        **model_kwargs,
    )

    optimizer = get_optimizer_cls(optimizer)
    training_loop = get_training_loop_cls(training_loop)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer_instance = optimizer(
        params=model_instance.get_grad_params(),
        **optimizer_kwargs,
    )

    if negative_sampler is None:
        training_loop_instance: TrainingLoop = training_loop(
            model=model_instance,
            optimizer=optimizer_instance,
        )
    elif training_loop is not OWATrainingLoop:
        raise ValueError('Can not specify negative sampler with CWA')
    else:
        negative_sampler = get_negative_sampler_cls(negative_sampler)
        training_loop_instance: TrainingLoop = training_loop(
            model=model_instance,
            optimizer=optimizer_instance,
            negative_sampler_cls=negative_sampler,
        )

    evaluator = get_evaluator_cls(evaluator)
    evaluator_instance: Evaluator = evaluator(
        **(evaluator_kwargs or {}),
    )

    # Early stopping
    if early_stopping:
        if early_stopping_kwargs is None:
            early_stopping_kwargs = {}
        if validation_triples_factory is None:
            raise ValueError('Must specify a validation_triples_factory or a dataset for using early stopping.')
        early_stopper = EarlyStopper(
            model=model_instance,
            evaluator=evaluator_instance,
            evaluation_triples_factory=validation_triples_factory,
            **early_stopping_kwargs,
        )
    else:
        early_stopper = None

    if training_kwargs is None:
        training_kwargs = {}
    training_kwargs.setdefault('num_epochs', 5)
    training_kwargs.setdefault('batch_size', 256)
    training_kwargs.setdefault('early_stopper', early_stopper)

    # Train like Cristiano Ronaldo
    losses = training_loop_instance.train(**training_kwargs)

    # Evaluate
    metric_results: MetricResults = evaluator_instance.evaluate(
        model=model_instance,
        mapped_triples=testing_triples_factory.mapped_triples,
        **(evaluation_kwargs or {}),
    )

    return PipelineResult(
        model=model_instance,
        training_loop=training_loop_instance,
        losses=losses,
        metric_results=metric_results,
    )
