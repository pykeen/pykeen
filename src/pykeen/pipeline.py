# -*- coding: utf-8 -*-

"""The easiest way to train and evaluate a model is with the :func:`pykeen.pipeline.pipeline` function.

It provides a high-level entry point into the extensible functionality of
this package.

Training a Model
~~~~~~~~~~~~~~~~
The following example shows how to train and evaluate the :class:`pykeen.models.TransE` model
on the :class:`pykeen.dataset.Nations` dataset. Throughout the documentation, you'll notice
that each asset has a corresponding class in PyKEEN. You can follow the links to learn more
about each and see the reference on how to use them specifically. Don't worry, in this part of
the tutorial, the :func:`pykeen.pipeline.pipeline` function will take care of everything for you.

>>> from pykeen.pipeline import pipeline
>>> result = pipeline(
...     dataset='Nations',
...     model='TransE',
... )
>>> result.save_to_directory('nations_transe')

The results are returned in a :class:`pykeen.pipeline.PipelineResult` instance, which has
attributes for the trained model, the training loop, and the evaluation.

In this example, the model was given as a string. A list of available models can be found in
:mod:`pykeen.models`. Alternatively, the class corresponding to the implementation of the model
could be used as in:

>>> from pykeen.pipeline import pipeline
>>> from pykeen.models import TransE
>>> result = pipeline(
...     dataset='Nations',
...     model=TransE,
... )
>>> result.save_to_directory('nations_transe')

In this example, the data set was given as a string. A list of available data sets can be found in
:mod:`pykeen.datasets`. Alternatively, the instance of the :class:`pykeen.datasets.DataSet` could be
used as in:

>>> from pykeen.pipeline import pipeline
>>> from pykeen.models import TransE
>>> from pykeen.datasets import Nations
>>> result = pipeline(
...     dataset=Nations,
...     model=TransE,
... )
>>> result.save_to_directory('nations_transe')

In each of the previous three examples, the training approach, optimizer, and evaluation scheme
were omitted. By default, the stochastic local closed world assumption (sLCWA) training approach is used in training.
This can be explicitly given as a string:

>>> from pykeen.pipeline import pipeline
>>> result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     training_loop='sLCWA',
... )
>>> result.save_to_directory('nations_transe')

Alternatively, the local closed world assumption (LCWA) training approach can be given with ``'LCWA'``.
No additional configuration is necessary, but it's worth reading up on the differences between these training
approaches.

>>> from pykeen.pipeline import pipeline
>>> result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     training_loop='LCWA',
... )
>>> result.save_to_directory('nations_transe')

One of these differences is that the sLCWA relies on *negative sampling*. The type of negative sampling
can be given as in:

>>> from pykeen.pipeline import pipeline
>>> result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     training_loop='sLCWA',
...     negative_sampler='basic',
... )
>>> result.save_to_directory('nations_transe')

In this example, the negative sampler was given as a string. A list of available negative samplers
can be found in :mod:`pykeen.sampling`. Alternatively, the class corresponding to the implementation
of the negative sampler could be used as in:

>>> from pykeen.pipeline import pipeline
>>> from pykeen.sampling import BasicNegativeSampler
>>> result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     training_loop='sLCWA',
...     negative_sampler=BasicNegativeSampler,
... )
>>> result.save_to_directory('nations_transe')

.. warning ::

   The ``negative_sampler`` keyword argument should not be used if the LCWA is being used.
   In general, all other options are available under either training approach.

The type of evaluation perfomed can be specified with the ``evaluator`` keyword. By default,
rank-based evaluation is used. It can be given explictly as in:

>>> from pykeen.pipeline import pipeline
>>> result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     evaluator='RankBasedEvaluator',
... )
>>> result.save_to_directory('nations_transe')

In this example, the evaluator string. A list of available evaluators can be found in
:mod:`pykeen.evaluation`. Alternatively, the class corresponding to the implementation
of the evaluator could be used as in:

>>> from pykeen.pipeline import pipeline
>>> from pykeen.evaluation import RankBasedEvaluator
>>> result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     evaluator=RankBasedEvaluator,
... )
>>> result.save_to_directory('nations_transe')

PyKEEN implements early stopping, which can be turned on with the ``stopper`` keyword
argument as in:

>>> from pykeen.pipeline import pipeline
>>> result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     stopper='early',
... )
>>> result.save_to_directory('nations_transe')

Deeper Configuration
~~~~~~~~~~~~~~~~~~~~
Arguments for the model can be given as a dictionary using ``model_kwargs``.

>>> from pykeen.pipeline import pipeline
>>> pipeline_result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     model_kwargs=dict(
...         scoring_fct_norm=2,
...     ),
... )
>>> result.save_to_directory('nations_transe')

The entries in ``model_kwargs`` correspond to the arguments given to :func:`pykeen.models.TransE.__init__`. For a
complete listing of models, see :mod:`pykeen.models`, where there are links to the reference for each
model that explain what kwargs are possible.

Because the pipeline takes care of looking up classes and instantiating them,
there are several other parameters to :func:`pykeen.pipeline.pipeline` that
can be used to specify the parameters during their respective instantiations.

Arguments can be given to the dataset with ``dataset_kwargs``. These are passed on to
the :class:`pykeen.dataset.Nations`

Bring Your Own Data
~~~~~~~~~~~~~~~~~~~
As an alternative to using a pre-packaged dataset, the training and testing can be set
explicitly with instances of :class:`pykeen.triples.TriplesFactory`. For convenience,
the default data sets are also provided as subclasses of :class:`pykeen.triples.TriplesFactory`.

.. warning ::

    Make sure they are mapped to the same entities.

>>> from pykeen.datasets import Nations
>>> from pykeen.triples import TriplesFactory
>>> from pykeen.pipeline import pipeline
>>> nations = Nations()
>>> training: TriplesFactory = nations.training
>>> testing: TriplesFactory = nations.testing
>>> pipeline_result = pipeline(
...     training_triples_factory=training,
...     testing_triples_factory=testing,
...     model='TransE',
... )
>>> result.save_to_directory('nations_transe')

.. todo:: Example with creation of triples factory
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, Union

import pandas as pd
import torch
from torch.optim.optimizer import Optimizer

from .datasets import get_dataset
from .datasets.base import DataSet
from .evaluation import Evaluator, MetricResults, get_evaluator_cls
from .losses import Loss, get_loss_cls
from .models import get_model_cls
from .models.base import Model
from .optimizers import get_optimizer_cls
from .regularizers import Regularizer, get_regularizer_cls
from .sampling import NegativeSampler, get_negative_sampler_cls
from .stoppers import EarlyStopper, Stopper, get_stopper_cls
from .trackers import MLFlowResultTracker, ResultTracker
from .training import SLCWATrainingLoop, TrainingLoop, get_training_loop_cls
from .triples import TriplesFactory
from .utils import NoRandomSeedNecessary, Result, fix_dataclass_init_docs, resolve_device, set_random_seed
from .version import get_git_hash, get_version

__all__ = [
    'PipelineResult',
    'pipeline_from_path',
    'pipeline_from_config',
    'replicate_pipeline_from_config',
    'replicate_pipeline_from_path',
    'pipeline',
]

logger = logging.getLogger(__name__)


@fix_dataclass_init_docs
@dataclass
class PipelineResult(Result):
    """A dataclass containing the results of running :func:`pykeen.pipeline.pipeline`."""

    #: The random seed used at the beginning of the pipeline
    random_seed: int

    #: The model trained by the pipeline
    model: Model

    #: The training loop used by the pipeline
    training_loop: TrainingLoop

    #: The losses during training
    losses: List[float]

    #: The results evaluated by the pipeline
    metric_results: MetricResults

    #: How long in seconds did training take?
    train_seconds: float

    #: How long in seconds did evaluation take?
    evaluate_seconds: float

    #: An early stopper
    stopper: Optional[Stopper] = None

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

    def _get_results(self) -> Mapping[str, Any]:
        results = dict(
            times=dict(
                training=self.train_seconds,
                evaluation=self.evaluate_seconds,
            ),
            metrics=self.metric_results.to_dict(),
            losses=self.losses,
        )
        if self.stopper is not None and isinstance(self.stopper, EarlyStopper):
            results['stopper'] = self.stopper.get_summary_dict()
        return results

    def save_to_directory(self, directory: str, save_metadata: bool = True, save_replicates: bool = True) -> None:
        """Save all artifacts in the given directory."""
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, 'metadata.json'), 'w') as file:
            json.dump(self.metadata, file, indent=2, sort_keys=True)
        with open(os.path.join(directory, 'results.json'), 'w') as file:
            json.dump(self._get_results(), file, indent=2, sort_keys=True)
        if save_replicates:
            self.save_model(os.path.join(directory, 'trained_model.pkl'))


def replicate_pipeline_from_path(
    path: str,
    directory: str,
    replicates: int,
    move_to_cpu: bool = False,
    save_replicates: bool = True,
    **kwargs,
) -> None:
    """Run the same pipeline several times from a configuration file by path.

    :param path: The path to the JSON configuration for the experiment.
    :param directory: The output directory
    :param replicates: The number of replicates to run.
    :param move_to_cpu: Should the model be moved back to the CPU? Only relevant if training on GPU.
    :param save_replicates: Should the artifacts of the replicates be saved?
    """
    pipeline_results = (
        pipeline_from_path(path, **kwargs)
        for _ in range(replicates)
    )
    save_pipeline_results_to_directory(
        directory=directory,
        pipeline_results=pipeline_results,
        move_to_cpu=move_to_cpu,
        save_replicates=save_replicates,
    )


def replicate_pipeline_from_config(
    config: Mapping[str, Any],
    directory: str,
    replicates: int,
    move_to_cpu: bool = False,
    save_replicates: bool = True,
    **kwargs,
) -> None:
    """Run the same pipeline several times from a configuration dictionary.

    :param config: The configuration dictionary for the experiment.
    :param directory: The output directory
    :param replicates: The number of replicates to run
    :param move_to_cpu: Should the models be moved back to the CPU? Only relevant if training on GPU.
    :param save_replicates: Should the artifacts of the replicates be saved?
    """
    pipeline_results = (
        pipeline_from_config(config, **kwargs)
        for _ in range(replicates)
    )
    save_pipeline_results_to_directory(
        directory=directory,
        pipeline_results=pipeline_results,
        move_to_cpu=move_to_cpu,
        save_replicates=save_replicates,
    )


def _iterate_moved(pipeline_results: Iterable[PipelineResult]):
    for pipeline_result in pipeline_results:
        pipeline_result.model.to_cpu_()
        yield pipeline_result


def save_pipeline_results_to_directory(
    *,
    directory: str,
    pipeline_results: Iterable[PipelineResult],
    move_to_cpu: bool = False,
    save_metadata: bool = False,
    save_replicates: bool = True,
    width: int = 5,
) -> None:
    """Save the result set to the directory.

    :param directory: The directory in which the replicates will be saved
    :param pipeline_results: An iterable over results from training and evaluation
    :param move_to_cpu: Should the model be moved back to the CPU? Only relevant if training on GPU.
    :param save_metadata: Should the metadata be saved? Might be redundant in a scenario when you're
     using this function, so defaults to false.
    :param save_replicates: Should the artifacts of the replicates be saved?
    :param width: How many leading zeros should be put in the replicate names?
    """
    replicates_directory = os.path.join(directory, 'replicates')
    losses_rows = []

    if move_to_cpu:
        pipeline_results = _iterate_moved(pipeline_results)

    for i, pipeline_result in enumerate(pipeline_results):
        sd = os.path.join(replicates_directory, f'replicate-{i:0{width}}')
        os.makedirs(sd, exist_ok=True)
        pipeline_result.save_to_directory(sd, save_metadata=save_metadata, save_replicates=save_replicates)
        for epoch, loss in enumerate(pipeline_result.losses):
            losses_rows.append((i, epoch, loss))

    losses_df = pd.DataFrame(losses_rows, columns=['Replicate', 'Epoch', 'Loss'])
    losses_df.to_csv(os.path.join(directory, 'all_replicates_losses.tsv'), sep='\t', index=False)


def pipeline_from_path(
    path: str,
    mlflow_tracking_uri: Optional[str] = None,
    **kwargs,
) -> PipelineResult:
    """Run the pipeline with configuration in a JSON file at the given path.

    :param path: The path to an experiment JSON file
    :param mlflow_tracking_uri: The URL of the MLFlow tracking server. If None, do not use MLFlow for result tracking.
    """
    with open(path) as file:
        config = json.load(file)
    return pipeline_from_config(
        config=config,
        mlflow_tracking_uri=mlflow_tracking_uri,
        **kwargs,
    )


def pipeline_from_config(
    config: Mapping[str, Any],
    mlflow_tracking_uri: Optional[str] = None,
    **kwargs,
) -> PipelineResult:
    """Run the pipeline with a configuration dictionary.

    :param config: The experiment configuration dictionary
    :param mlflow_tracking_uri: The URL of the MLFlow tracking server. If None, do not use MLFlow for result tracking.
    """
    metadata, pipeline_kwargs = config['metadata'], config['pipeline']
    title = metadata.get('title')
    if title is not None:
        logger.info(f'Running: {title}')

    return pipeline(
        mlflow_tracking_uri=mlflow_tracking_uri,
        metadata=metadata,
        **pipeline_kwargs,
        **kwargs,
    )


def pipeline(  # noqa: C901
    *,
    # 1. Dataset
    dataset: Union[None, str, Type[DataSet]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training_triples_factory: Optional[TriplesFactory] = None,
    testing_triples_factory: Optional[TriplesFactory] = None,
    validation_triples_factory: Optional[TriplesFactory] = None,
    # 2. Model
    model: Union[str, Type[Model]],
    model_kwargs: Optional[Mapping[str, Any]] = None,
    # 3. Loss
    loss: Union[None, str, Type[Loss]] = None,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    # 4. Regularizer
    regularizer: Union[None, str, Type[Regularizer]] = None,
    regularizer_kwargs: Optional[Mapping[str, Any]] = None,
    # 5. Optimizer
    optimizer: Union[None, str, Type[Optimizer]] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    clear_optimizer: bool = True,
    # 6. Training Loop
    training_loop: Union[None, str, Type[TrainingLoop]] = None,
    negative_sampler: Union[None, str, Type[NegativeSampler]] = None,
    negative_sampler_kwargs: Optional[Mapping[str, Any]] = None,
    # 7. Training (ronaldo style)
    training_kwargs: Optional[Mapping[str, Any]] = None,
    stopper: Union[None, str, Type[Stopper]] = None,
    stopper_kwargs: Optional[Mapping[str, Any]] = None,
    # 8. Evaluation
    evaluator: Union[None, str, Type[Evaluator]] = None,
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    # Misc
    mlflow_tracking_uri: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    device: Union[None, str, torch.device] = None,
    random_seed: Optional[int] = None,
    use_testing_data: bool = True,
) -> PipelineResult:
    """Train and evaluate a model.

    :param dataset:
        The name of the dataset (a key from :data:`pykeen.datasets.datasets`) or the :class:`pykeen.datasets.DataSet`
        instance. Alternatively, the ``training_triples_factory`` and ``testing_triples_factory`` can be specified.
    :param dataset_kwargs:
        The keyword arguments passed to the dataset upon instantiation
    :param training_triples_factory:
        A triples factory with training instances if a a dataset was not specified
    :param testing_triples_factory:
        A triples factory with training instances if a dataset was not specified
    :param validation_triples_factory:
        A triples factory with validation instances if a dataset was not specified

    :param model:
        The name of the model or the model class
    :param model_kwargs:
        Keyword arguments to pass to the model class on instantiation

    :param loss:
        The name of the loss or the loss class.
    :param loss_kwargs:
        Keyword arguments to pass to the loss on instantiation

    :param regularizer:
        The name of the regularizer or the regularizer class.
    :param regularizer_kwargs:
        Keyword arguments to pass to the regularizer on instantiation

    :param optimizer:
        The name of the optimizer or the optimizer class. Defaults to :class:`torch.optim.Adagrad`.
    :param optimizer_kwargs:
        Keyword arguments to pass to the optimizer on instantiation
    :param clear_optimizer:
        Whether to delete the optimizer instance after training. As the optimizer might have additional memory
        consumption due to e.g. moments in Adam, this is the default option. If you want to continue training, you
        should set it to False, as the optimizer's internal parameter will get lost otherwise.

    :param training_loop:
        The name of the training loop's training approach (``'slcwa'`` or ``'lcwa'``) or the training loop class.
        Defaults to :class:`pykeen.training.SLCWATrainingLoop`.
    :param negative_sampler:
        The name of the negative sampler (``'basic'`` or ``'bernoulli'``) or the negative sampler class.
        Only allowed when training with sLCWA.
        Defaults to :class:`pykeen.sampling.BasicNegativeSampler`.
    :param negative_sampler_kwargs:
        Keyword arguments to pass to the negative sampler class on instantiation

    :param training_kwargs:
        Keyword arguments to pass to the training loop's train function on call
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

    :param mlflow_tracking_uri:
        The MLFlow tracking URL. If None is given, MLFlow is not used to track results.
    :param metadata: A JSON dictionary to store with the experiment
    :param use_testing_data: If true, use the testing triples. Otherwise, use the validation triples.
     Defaults to true - use testing triples.
    """
    if random_seed is None:
        random_seed = random.randint(0, 2 ** 32 - 1)
        logger.warning(f'No random seed is specified. Setting to {random_seed}.')
    set_random_seed(random_seed)

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

    result_tracker.log_params({'dataset': dataset})

    training_triples_factory, testing_triples_factory, validation_triples_factory = get_dataset(
        dataset=dataset,
        dataset_kwargs=dataset_kwargs,
        training_triples_factory=training_triples_factory,
        testing_triples_factory=testing_triples_factory,
        validation_triples_factory=validation_triples_factory,
    )

    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs.update(preferred_device=device)
    model_kwargs.setdefault('random_seed', NoRandomSeedNecessary)

    if regularizer is not None:
        # FIXME this should never happen.
        if 'regularizer' in model_kwargs:
            logger.warning('Can not specify regularizer in kwargs and model_kwargs. removing from model_kwargs')
            del model_kwargs['regularizer']
        regularizer_cls: Type[Regularizer] = get_regularizer_cls(regularizer)
        model_kwargs['regularizer'] = regularizer_cls(
            device=device,
            **(regularizer_kwargs or {}),
        )

    if loss is not None:
        if 'loss' in model_kwargs:  # FIXME
            logger.warning('duplicate loss in kwargs and model_kwargs. removing from model_kwargs')
            del model_kwargs['loss']
        loss_cls = get_loss_cls(loss)
        _loss = loss_cls(**(loss_kwargs or {}))
        model_kwargs.setdefault('loss', _loss)

    # Log model parameters
    result_tracker.log_params(model_kwargs, prefix='model')

    model = get_model_cls(model)
    model_instance: Model = model(
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
    elif training_loop is not SLCWATrainingLoop:
        raise ValueError('Can not specify negative sampler with LCWA')
    else:
        negative_sampler = get_negative_sampler_cls(negative_sampler)
        training_loop_instance: TrainingLoop = SLCWATrainingLoop(
            model=model_instance,
            optimizer=optimizer_instance,
            negative_sampler_cls=negative_sampler,
            negative_sampler_kwargs=negative_sampler_kwargs,
        )

    evaluator = get_evaluator_cls(evaluator)
    evaluator_instance: Evaluator = evaluator(
        **(evaluator_kwargs or {}),
    )

    if evaluation_kwargs is None:
        evaluation_kwargs = {}

    if training_kwargs is None:
        training_kwargs = {}

    # Stopping
    if 'stopper' in training_kwargs and stopper is not None:
        raise ValueError('Specified stopper in training_kwargs and as stopper')
    if 'stopper' in training_kwargs:
        stopper = training_kwargs.pop('stopper')
    if stopper_kwargs is None:
        stopper_kwargs = {}

    # Load the evaluation batch size for the stopper, if it has been set
    _evaluation_batch_size = evaluation_kwargs.get('batch_size')
    if _evaluation_batch_size is not None:
        stopper_kwargs.setdefault('evaluation_batch_size', _evaluation_batch_size)

    # By default there's a stopper that does nothing interesting
    stopper_cls: Type[Stopper] = get_stopper_cls(stopper)
    stopper: Stopper = stopper_cls(
        model=model_instance,
        evaluator=evaluator_instance,
        evaluation_triples_factory=validation_triples_factory,
        result_tracker=result_tracker,
        **stopper_kwargs,
    )

    training_kwargs.setdefault('num_epochs', 5)
    training_kwargs.setdefault('batch_size', 256)

    # Add logging for debugging
    logging.debug("Run Pipeline based on following config:")
    logging.debug(f"dataset: {dataset}")
    logging.debug(f"dataset_kwargs: {dataset_kwargs}")
    logging.debug(f"model: {model}")
    logging.debug(f"model_kwargs: {model_kwargs}")
    logging.debug(f"loss: {loss}")
    logging.debug(f"loss_kwargs: {loss_kwargs}")
    logging.debug(f"regularizer: {regularizer}")
    logging.debug(f"regularizer_kwargs: {regularizer_kwargs}")
    logging.debug(f"optimizer: {optimizer}")
    logging.debug(f"optimizer_kwargs: {optimizer_kwargs}")
    logging.debug(f"training_loop: {training_loop}")
    logging.debug(f"negative_sampler: {negative_sampler}")
    logging.debug(f"_negative_sampler_kwargs: {negative_sampler_kwargs}")
    logging.debug(f"_training_kwargs: {training_kwargs}")
    logging.debug(f"stopper: {stopper}")
    logging.debug(f"stopper_kwargs: {stopper_kwargs}")
    logging.debug(f"evaluator: {evaluator}")
    logging.debug(f"evaluator_kwargs: {evaluator_kwargs}")

    # Train like Cristiano Ronaldo
    training_start_time = time.time()
    losses = training_loop_instance.train(
        stopper=stopper,
        result_tracker=result_tracker,
        clear_optimizer=clear_optimizer,
        **training_kwargs,
    )
    training_end_time = time.time() - training_start_time

    if use_testing_data:
        mapped_triples = testing_triples_factory.mapped_triples
    else:
        mapped_triples = validation_triples_factory.mapped_triples

    # Evaluate
    # Reuse optimal evaluation parameters from training if available
    if evaluator_instance.batch_size is not None or evaluator_instance.slice_size is not None:
        evaluation_kwargs['batch_size'] = evaluator_instance.batch_size
        evaluation_kwargs['slice_size'] = evaluator_instance.slice_size
    # Add logging about evaluator for debugging
    logging.debug("Evaluation will be run with following parameters:")
    logging.debug(f"evaluation_kwargs: {evaluation_kwargs}")
    evaluate_start_time = time.time()
    metric_results: MetricResults = evaluator_instance.evaluate(
        model=model_instance,
        mapped_triples=mapped_triples,
        **evaluation_kwargs,
    )
    evaluate_end_time = time.time() - evaluate_start_time
    result_tracker.log_metrics(
        metrics=metric_results.to_dict(),
        step=training_kwargs.get('num_epochs'),
    )
    result_tracker.end_run()

    return PipelineResult(
        random_seed=random_seed,
        model=model_instance,
        training_loop=training_loop_instance,
        losses=losses,
        stopper=stopper,
        metric_results=metric_results,
        metadata=metadata,
        train_seconds=training_end_time,
        evaluate_seconds=evaluate_end_time,
    )
