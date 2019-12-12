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

Alternatively, the local closed world assumption (LCWA) can be given with ``'LCWA'``. No additional configuration
is necessary, but it's worth reading up on the differences between these assumptions.

>>> from poem.pipeline import pipeline
>>> result = pipeline(
...     model='TransE',
...     data_set='Nations',
...     training_loop='LCWA',
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

   The ``negative_sampler`` keyword argument should not be used if the LCWA is being used.
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

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Type, Union

import pandas as pd
import torch
from torch.optim.optimizer import Optimizer

from .datasets import DataSet, get_dataset
from .evaluation import Evaluator, MetricResults, get_evaluator_cls
from .losses import Loss, get_loss_cls
from .models import get_model_cls
from .models.base import BaseModule
from .optimizers import get_optimizer_cls
from .regularizers import Regularizer, get_regularizer_cls
from .sampling import NegativeSampler, get_negative_sampler_cls
from .training import EarlyStopper, OWATrainingLoop, TrainingLoop, get_training_loop_cls
from .triples import TriplesFactory
from .utils import MLFlowResultTracker, ResultTracker, resolve_device
from .version import get_git_hash, get_version

__all__ = [
    'PipelineResult',
    'PipelineResultSet',
    'pipeline_from_path',
    'pipeline',
]

logger = logging.getLogger(__name__)


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

    #: The version of PyKEEN used to create these results
    version: str = field(default_factory=get_version)

    #: The git hash of PyKEEN used to create these results
    git_hash: str = field(default_factory=get_git_hash)

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

    def save_to_directory(self, directory: str) -> None:
        """Save all artifacts in the given directory."""
        with open(os.path.join(directory, 'losses.json'), 'w') as file:
            json.dump(self.losses, file, indent=2)
        with open(os.path.join(directory, 'metric_results.json'), 'w') as file:
            json.dump(self.metric_results.to_dict(), file, indent=2)
        with open(os.path.join(directory, 'configuration.json'), 'w') as file:
            json.dump(self._get_configuration(), file, indent=2)
        with open(os.path.join(directory, 'metadata.json'), 'w') as file:
            json.dump(self.metadata, file, indent=2)
        with open(os.path.join(directory, 'environment.json'), 'w') as file:
            json.dump(self._get_environment(), file, indent=2)
        self.save_model(os.path.join(directory, 'trained_model.pkl'))

    def _get_environment(self):
        return dict(
            pykeen=dict(
                version=self.version,
                git_hash=self.git_hash,
            ),
        )

    def _get_configuration(self) -> Mapping[str, Any]:
        """Get all of the configuration out of the model and training loop."""
        # FIXME
        return {}


@dataclass
class PipelineResultSet:
    """A set of results."""

    pipeline_results: List[PipelineResult]

    @classmethod
    def from_path(cls, path: str, replicates: int = 10) -> 'PipelineResultSet':
        """Run the same pipeline several times."""
        return cls([
            pipeline_from_path(path)
            for _ in range(replicates)
        ])

    def get_loss_df(self) -> pd.DataFrame:
        """Get the losses as a dataframe."""
        return pd.DataFrame(
            [
                (replicate, epoch, loss)
                for replicate, result in enumerate(self.pipeline_results, start=1)
                for epoch, loss in enumerate(result.losses, start=1)
            ],
            columns=['Replicate', 'Epoch', 'Loss'],
        )

    def plot_losses(self, sns_kwargs: Optional[Mapping[str, Any]] = None):
        """Plot the several losses."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = self.get_loss_df()
        sns.set()
        if self.pipeline_results[0].title is not None:
            plt.title(self.pipeline_results[0].title)
        return sns.lineplot(data=df, x='Epoch', y='Loss', **(sns_kwargs or {}))


def pipeline_from_path(
    path: str,
    mlflow_tracking_uri: Optional[str] = None,
) -> PipelineResult:
    """Run the pipeline with configuration in a JSON file at the given path.

    :param path: The path to an experiment JSON file
    :param mlflow_tracking_uri: The URL of the MLFlow tracking server. If None, do not use MLFlow for result tracking.
    """
    with open(path) as file:
        config = json.load(file)

    metadata, pipeline_kwargs = config['metadata'], config['pipeline']
    title = metadata.get('title')
    if title is not None:
        logger.info(f'Running: {title}')

    return pipeline(
        mlflow_tracking_uri=mlflow_tracking_uri,
        metadata=metadata,
        **pipeline_kwargs,
    )


def pipeline(  # noqa: C901
    *,
    model: Union[str, Type[BaseModule]],
    model_kwargs: Optional[Mapping[str, Any]] = None,
    optimizer: Union[None, str, Type[Optimizer]] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    loss: Union[None, str, Type[Loss]] = None,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    training_loop: Union[None, str, Type[TrainingLoop]] = None,
    data_set: Union[None, str, DataSet] = None,
    training_triples_factory: Optional[TriplesFactory] = None,
    testing_triples_factory: Optional[TriplesFactory] = None,
    validation_triples_factory: Optional[TriplesFactory] = None,
    triples_factory_kwargs: Optional[Mapping[str, Any]] = None,
    negative_sampler: Union[None, str, Type[NegativeSampler]] = None,
    negative_sampler_kwargs: Optional[Mapping[str, Any]] = None,
    training_kwargs: Optional[Mapping[str, Any]] = None,
    early_stopping: bool = False,
    early_stopping_kwargs: Optional[Mapping[str, Any]] = None,
    evaluator: Union[None, str, Type[Evaluator]] = None,
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    mlflow_tracking_uri: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    regularizer: Union[None, str, Type[Regularizer]] = None,
    regularizer_kwargs: Optional[Mapping[str, Any]] = None,
    device: Union[None, str, torch.device] = None,
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
    :param training_loop: The name of the training loop's assumption (``'owa'`` or ``'lcwa'``)
     or the training loop class. Defaults to :class:`poem.training.OWATrainingLoop`.
    :param negative_sampler: The name of the negative sampler (``'basic'`` or ``'bernoulli'``)
     or the negative sampler class. Only allowed when training with OWA. Defaults to
     :class:`poem.sampling.BasicNegativeSampler`.
    :param evaluator: The name of the evaluator or an evaluator class. Defaults to
     :class:`poem.evaluation.RankBasedEvaluator`.
    :param early_stopping: Whether to use early stopping. Defaults to false.
    :param negative_sampler_kwargs: Keyword arguments to pass to the negative sampler class on instantiation
    :param model_kwargs: Keyword arguments to pass to the model class on instantiation
    :param optimizer_kwargs: Keyword arguments to pass to the optimizer on instantiation
    :param training_kwargs: Keyword arguments to pass to the training loop's train
     function on call
    :param early_stopping_kwargs: Keyword arguments to pass to the EarlyStopper upon instantiation.
    :param evaluator_kwargs: Keyword arguments to pass to the evaluator on instantiation
    :param evaluation_kwargs: Keyword arguments to pass to the evaluator's evaluate
     function on call
    :param mlflow_tracking_uri:
        The MLFlow tracking URL. If None is given, MLFlow is not used to track results.
    :param metadata: A JSON dictionary to store with the experiment
    """
    # Create result store
    if mlflow_tracking_uri is not None:
        result_tracker = MLFlowResultTracker(tracking_uri=mlflow_tracking_uri)
    else:
        result_tracker = ResultTracker()

    if not metadata:
        metadata = {}
    title = metadata.get('title')

    # Start tracking
    result_tracker.start_run(run_name=title)

    device = resolve_device(device)

    result_tracker.log_params({'dataset': data_set})
    training_triples_factory, testing_triples_factory, validation_triples_factory = get_dataset(
        dataset=data_set,
        training_triples_factory=training_triples_factory,
        testing_triples_factory=testing_triples_factory,
        validation_triples_factory=validation_triples_factory,
        triples_factory_kwargs=triples_factory_kwargs,
    )

    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs.update(preferred_device=device)

    if regularizer is not None and 'regularizer' in model_kwargs:
        raise ValueError('Can not specify regularizer in kwargs and model_kwargs')
    elif regularizer is not None:
        regularizer_cls: Type[Regularizer] = get_regularizer_cls(regularizer)
        model_kwargs['regularizer'] = regularizer_cls(
            device=device,
            **(regularizer_kwargs or {}),
        )

    if loss is not None:
        loss_cls = get_loss_cls(loss)
        _loss = loss_cls(**(loss_kwargs or {}))
        model_kwargs.setdefault('loss', _loss)

    # Log model parameters
    result_tracker.log_params(model_kwargs, prefix='model')

    model = get_model_cls(model)
    model_instance: BaseModule = model(
        triples_factory=training_triples_factory,
        **model_kwargs,
    )

    optimizer = get_optimizer_cls(optimizer)
    training_loop = get_training_loop_cls(training_loop)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    # Log optimizer parameters
    result_tracker.log_params({'class': optimizer, 'kwargs': optimizer_kwargs}, prefix='optimizer')
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
        raise ValueError('Can not specify negative sampler with LCWA')
    else:
        negative_sampler = get_negative_sampler_cls(negative_sampler)
        training_loop_instance: TrainingLoop = OWATrainingLoop(
            model=model_instance,
            optimizer=optimizer_instance,
            negative_sampler_cls=negative_sampler,
            negative_sampler_kwargs=negative_sampler_kwargs,
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
            result_tracker=result_tracker,
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
    losses = training_loop_instance.train(**training_kwargs, result_tracker=result_tracker)

    # Evaluate
    metric_results: MetricResults = evaluator_instance.evaluate(
        model=model_instance,
        mapped_triples=testing_triples_factory.mapped_triples,
        **(evaluation_kwargs or {}),
    )
    result_tracker.log_metrics(
        metrics=metric_results.to_dict(),
        step=training_kwargs.get('num_epochs'),
    )
    result_tracker.end_run()

    return PipelineResult(
        model=model_instance,
        training_loop=training_loop_instance,
        losses=losses,
        metric_results=metric_results,
        metadata=metadata,
    )
