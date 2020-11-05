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
>>> pipeline_result = pipeline(
...     dataset='Nations',
...     model='TransE',
... )
>>> pipeline_result.save_to_directory('nations_transe')

The results are returned in a :class:`pykeen.pipeline.PipelineResult` instance, which has
attributes for the trained model, the training loop, and the evaluation.

In this example, the model was given as a string. A list of available models can be found in
:mod:`pykeen.models`. Alternatively, the class corresponding to the implementation of the model
could be used as in:

>>> from pykeen.pipeline import pipeline
>>> from pykeen.models import TransE
>>> pipeline_result = pipeline(
...     dataset='Nations',
...     model=TransE,
... )
>>> pipeline_result.save_to_directory('nations_transe')

In this example, the data set was given as a string. A list of available data sets can be found in
:mod:`pykeen.datasets`. Alternatively, the instance of the :class:`pykeen.datasets.DataSet` could be
used as in:

>>> from pykeen.pipeline import pipeline
>>> from pykeen.models import TransE
>>> from pykeen.datasets import Nations
>>> pipeline_result = pipeline(
...     dataset=Nations,
...     model=TransE,
... )
>>> pipeline_result.save_to_directory('nations_transe')

In each of the previous three examples, the training approach, optimizer, and evaluation scheme
were omitted. By default, the stochastic local closed world assumption (sLCWA) training approach is used in training.
This can be explicitly given as a string:

>>> from pykeen.pipeline import pipeline
>>> pipeline_result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     training_loop='sLCWA',
... )
>>> pipeline_result.save_to_directory('nations_transe')

Alternatively, the local closed world assumption (LCWA) training approach can be given with ``'LCWA'``.
No additional configuration is necessary, but it's worth reading up on the differences between these training
approaches.

>>> from pykeen.pipeline import pipeline
>>> pipeline_result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     training_loop='LCWA',
... )
>>> pipeline_result.save_to_directory('nations_transe')

One of these differences is that the sLCWA relies on *negative sampling*. The type of negative sampling
can be given as in:

>>> from pykeen.pipeline import pipeline
>>> pipeline_result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     training_loop='sLCWA',
...     negative_sampler='basic',
... )
>>> pipeline_result.save_to_directory('nations_transe')

In this example, the negative sampler was given as a string. A list of available negative samplers
can be found in :mod:`pykeen.sampling`. Alternatively, the class corresponding to the implementation
of the negative sampler could be used as in:

>>> from pykeen.pipeline import pipeline
>>> from pykeen.sampling import BasicNegativeSampler
>>> pipeline_result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     training_loop='sLCWA',
...     negative_sampler=BasicNegativeSampler,
... )
>>> pipeline_result.save_to_directory('nations_transe')

.. warning ::

   The ``negative_sampler`` keyword argument should not be used if the LCWA is being used.
   In general, all other options are available under either training approach.

The type of evaluation perfomed can be specified with the ``evaluator`` keyword. By default,
rank-based evaluation is used. It can be given explictly as in:

>>> from pykeen.pipeline import pipeline
>>> pipeline_result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     evaluator='RankBasedEvaluator',
... )
>>> pipeline_result.save_to_directory('nations_transe')

In this example, the evaluator string. A list of available evaluators can be found in
:mod:`pykeen.evaluation`. Alternatively, the class corresponding to the implementation
of the evaluator could be used as in:

>>> from pykeen.pipeline import pipeline
>>> from pykeen.evaluation import RankBasedEvaluator
>>> pipeline_result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     evaluator=RankBasedEvaluator,
... )
>>> pipeline_result.save_to_directory('nations_transe')

PyKEEN implements early stopping, which can be turned on with the ``stopper`` keyword
argument as in:

>>> from pykeen.pipeline import pipeline
>>> pipeline_result = pipeline(
...     dataset='Nations',
...     model='TransE',
...     stopper='early',
... )
>>> pipeline_result.save_to_directory('nations_transe')

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
>>> pipeline_result.save_to_directory('nations_transe')

The entries in ``model_kwargs`` correspond to the arguments given to :func:`pykeen.models.TransE.__init__`. For a
complete listing of models, see :mod:`pykeen.models`, where there are links to the reference for each
model that explain what kwargs are possible.

Because the pipeline takes care of looking up classes and instantiating them,
there are several other parameters to :func:`pykeen.pipeline.pipeline` that
can be used to specify the parameters during their respective instantiations.

Arguments can be given to the dataset with ``dataset_kwargs``. These are passed on to
the :class:`pykeen.dataset.Nations`
"""

import ftplib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Set, Type, Union

import pandas as pd
import torch
from torch.optim.optimizer import Optimizer

from .datasets import get_dataset
from .datasets.base import DataSet
from .evaluation import Evaluator, MetricResults, get_evaluator_cls
from .losses import Loss, _LOSS_SUFFIX, get_loss_cls
from .models import get_model_cls
from .models.base import Model
from .optimizers import get_optimizer_cls
from .regularizers import Regularizer, get_regularizer_cls
from .sampling import NegativeSampler, get_negative_sampler_cls
from .stoppers import EarlyStopper, Stopper, get_stopper_cls
from .trackers import ResultTracker, get_result_tracker_cls
from .training import SLCWATrainingLoop, TrainingLoop, get_training_loop_cls
from .triples import TriplesFactory
from .utils import (
    Result, ensure_ftp_directory, fix_dataclass_init_docs, get_json_bytes_io, get_model_io, normalize_string,
    random_non_negative_int, resolve_device, set_random_seed,
)
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

REDUCER_RELATION_WHITELIST = {'PCA'}


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

    def plot_losses(self, ax=None):
        """Plot the losses per epoch."""
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        import seaborn as sns
        sns.set_style('darkgrid')

        rv = sns.lineplot(x=range(len(self.losses)), y=self.losses, ax=ax)

        loss_name = normalize_string(self.model.loss.__class__.__name__, suffix=_LOSS_SUFFIX)
        ax.set_ylabel(f'{loss_name} Loss')
        ax.set_xlabel('Epoch')
        ax.set_title(self.title if self.title is not None else 'Losses Plot')
        return rv

    def plot_er(  # noqa: C901
        self,
        model: Optional[str] = None,
        margin: float = 0.4,
        ax=None,
        entities: Optional[Set[str]] = None,
        relations: Optional[Set[str]] = None,
        apply_limits: bool = True,
        plot_entities: bool = True,
        plot_relations: Optional[bool] = None,
        annotation_x_offset: float = 0.02,
        annotation_y_offset: float = 0.03,
        **kwargs,
    ):
        """Plot the reduced entities and relation vectors in 2D.

        :param model: The dimensionality reduction model from :mod:`sklearn`. Defaults to PCA.
            Can also use KPCA, GRP, SRP, TSNE, LLE, ISOMAP, MDS, or SE.
        :param kwargs: The keyword arguments passed to `__init__()` of
            the reducer class (e.g., PCA, TSNE)
        :param plot_relations: By default, this is only enabled on translational distance models
            like :class:`pykeen.models.TransE`.

        .. warning::

            Plotting relations and entities on the same plot is only
            meaningful for translational distance models like TransE.
        """
        if not plot_entities and not plot_relations:
            raise ValueError

        if plot_relations is None:  # automatically set to true for translational models, false otherwise
            plot_relations = self.model.__class__.__name__.lower().startswith('trans')

        if model is None:
            model = 'PCA'
        reducer_cls, reducer_kwargs = _get_reducer_cls(model, **kwargs)
        if plot_relations and reducer_cls.__name__ not in REDUCER_RELATION_WHITELIST:
            raise ValueError(f'Can not use reducer {reducer_cls} when projecting relations. Will result in nonsense')
        reducer = reducer_cls(n_components=2, **reducer_kwargs)

        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        import seaborn as sns
        sns.set_style('whitegrid')

        if plot_relations and plot_entities:
            e_embeddings, e_reduced = _reduce_embeddings(self.model.entity_embeddings, reducer, fit=True)
            r_embeddings, r_reduced = _reduce_embeddings(self.model.relation_embeddings, reducer, fit=False)

            xmax = max(r_embeddings[:, 0].max(), e_embeddings[:, 0].max()) + margin
            xmin = min(r_embeddings[:, 0].min(), e_embeddings[:, 0].min()) - margin
            ymax = max(r_embeddings[:, 1].max(), e_embeddings[:, 1].max()) + margin
            ymin = min(r_embeddings[:, 1].min(), e_embeddings[:, 1].min()) - margin
        elif plot_relations:
            e_embeddings, e_reduced = None, False
            r_embeddings, r_reduced = _reduce_embeddings(self.model.relation_embeddings, reducer, fit=True)

            xmax = r_embeddings[:, 0].max() + margin
            xmin = r_embeddings[:, 0].min() - margin
            ymax = r_embeddings[:, 1].max() + margin
            ymin = r_embeddings[:, 1].min() - margin
        elif plot_entities:
            e_embeddings, e_reduced = _reduce_embeddings(self.model.entity_embeddings, reducer, fit=True)
            r_embeddings, r_reduced = None, False

            xmax = e_embeddings[:, 0].max() + margin
            xmin = e_embeddings[:, 0].min() - margin
            ymax = e_embeddings[:, 1].max() + margin
            ymin = e_embeddings[:, 1].min() - margin
        else:
            raise ValueError  # not even possible

        if not e_reduced and not r_reduced:
            subtitle = ''
        elif reducer_kwargs:
            subtitle = ", ".join("=".join(item) for item in reducer_kwargs.items())
            subtitle = f' using {reducer_cls.__name__} ({subtitle})'
        else:
            subtitle = f' using {reducer_cls.__name__}'

        if plot_entities:
            entity_id_to_label = self.model.triples_factory.entity_id_to_label
            for entity_id, entity_reduced_embedding in enumerate(e_embeddings):
                entity_label = entity_id_to_label[entity_id]
                if entities and entity_label not in entities:
                    continue
                x, y = entity_reduced_embedding
                ax.scatter(x, y, color='black')
                ax.annotate(entity_label, (x + annotation_x_offset, y + annotation_y_offset))

        if plot_relations:
            relation_id_to_label = self.model.triples_factory.relation_id_to_label
            for relation_id, relation_reduced_embedding in enumerate(r_embeddings):
                relation_label = relation_id_to_label[relation_id]
                if relations and relation_label not in relations:
                    continue
                x, y = relation_reduced_embedding
                ax.arrow(0, 0, x, y, color='black')
                ax.annotate(relation_label, (x + annotation_x_offset, y + annotation_y_offset))

        if plot_entities and plot_relations:
            ax.set_title(f'Entity/Relation Plot{subtitle}')
        elif plot_entities:
            ax.set_title(f'Entity Plot{subtitle}')
        elif plot_relations:
            ax.set_title(f'Relation Plot{subtitle}')

        if apply_limits:
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

        return ax

    def plot(self, er_kwargs: Optional[Mapping[str, str]] = None, figsize=(10, 4)):
        """Plot all plots."""
        import matplotlib.pyplot as plt
        fig, (lax, rax) = plt.subplots(1, 2, figsize=figsize)

        self.plot_losses(ax=lax)
        self.plot_er(ax=rax, **(er_kwargs or {}))

        plt.tight_layout()

    def save_model(self, path: str) -> None:
        """Save the trained model to the given path using :func:`torch.save`.

        :param path: The path to which the model is saved. Should have an extension appropriate for a pickle,
         like `*.pkl` or `*.pickle`.

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

    def save_to_ftp(self, directory: str, ftp: ftplib.FTP) -> None:
        """Save all artifacts to the given directory in the FTP server.

        :param directory: The directory in the FTP server to save to
        :param ftp: A connection to the FTP server

        The following code will train a model and upload it to FTP using Python's builtin
        :class:`ftplib.FTP`:

        .. code-block:: python

            import ftplib
            from pykeen.pipeline import pipeline

            directory = 'test/test'
            pipeline_result = pipeline(
                model='TransE',
                dataset='Kinships',
            )
            with ftplib.FTP(host='0.0.0.0', user='user', passwd='12345') as ftp:
                pipeline_result.save_to_ftp(directory, ftp)

        If you want to try this with your own local server, run this code based on the
        example from Giampaolo Rodola's excellent library,
        `pyftpdlib <https://github.com/giampaolo/pyftpdlib#quick-start>`_.

        .. code-block:: python

            import os
            from pyftpdlib.authorizers import DummyAuthorizer
            from pyftpdlib.handlers import FTPHandler
            from pyftpdlib.servers import FTPServer

            authorizer = DummyAuthorizer()
            authorizer.add_user("user", "12345", homedir=os.path.expanduser('~/ftp'), perm="elradfmwMT")

            handler = FTPHandler
            handler.authorizer = authorizer

            address = '0.0.0.0', 21
            server = FTPServer(address, handler)
            server.serve_forever()
        """
        ensure_ftp_directory(ftp=ftp, directory=directory)

        metadata_path = os.path.join(directory, 'metadata.json')
        ftp.storbinary(f'STOR {metadata_path}', get_json_bytes_io(self.metadata))

        results_path = os.path.join(directory, 'results.json')
        ftp.storbinary(f'STOR {results_path}', get_json_bytes_io(self._get_results()))

        model_path = os.path.join(directory, 'trained_model.pkl')
        ftp.storbinary(f'STOR {model_path}', get_model_io(self.model))

    def save_to_s3(self, directory: str, bucket: str, s3=None) -> None:
        """Save all artifacts to the given directory in an S3 Bucket.

        :param directory: The directory in the S3 bucket
        :param bucket: The name of the S3 bucket
        :param s3: A client from :func:`boto3.client`, if already instantiated

        .. note:: Need to have ``~/.aws/credentials`` file set up. Read: https://realpython.com/python-boto3-aws-s3/

        The following code will train a model and upload it to S3 using :mod:`boto3`:

        .. code-block:: python

            import time
            from pykeen.pipeline import pipeline
            pipeline_result = pipeline(
                dataset='Kinships',
                model='TransE',
            )
            directory = f'tests/{time.strftime("%Y-%m-%d-%H%M%S")}'
            bucket = 'pykeen'
            pipeline_result.save_to_s3(directory, bucket=bucket)
        """
        if s3 is None:
            import boto3
            s3 = boto3.client('s3')

        metadata_path = os.path.join(directory, 'metadata.json')
        s3.upload_fileobj(get_json_bytes_io(self.metadata), bucket, metadata_path)

        results_path = os.path.join(directory, 'results.json')
        s3.upload_fileobj(get_json_bytes_io(self._get_results()), bucket, results_path)

        model_path = os.path.join(directory, 'trained_model.pkl')
        s3.upload_fileobj(get_model_io(self.model), bucket, model_path)


def _reduce_embeddings(embeddings, reducer, fit: bool = False):
    embeddings_numpy = embeddings.weight.detach().numpy()
    if embeddings_numpy.shape[1] == 2:
        logger.debug('not reducing entity embeddings, already dim=2')
        return embeddings_numpy, False
    elif fit:
        return reducer.fit_transform(embeddings_numpy), True
    else:
        return reducer.transform(embeddings_numpy), True


def _get_reducer_cls(model: str, **kwargs):
    """Get the model class by name and default kwargs.

    :param model: The name of the model. Can choose from: PCA, KPCA, GRP,
        SRP, TSNE, LLE, ISOMAP, MDS, or SE.
    :param kwargs:
    :return:
    """
    if model.upper() == 'PCA':
        from sklearn.decomposition import PCA as Reducer  # noqa:N811
    elif model.upper() == 'KPCA':
        kwargs.setdefault('kernel', 'rbf')
        from sklearn.decomposition import KernelPCA as Reducer
    elif model.upper() == 'GRP':
        from sklearn.random_projection import GaussianRandomProjection as Reducer
    elif model.upper() == 'SRP':
        from sklearn.random_projection import SparseRandomProjection as Reducer
    elif model.upper() in {'T-SNE', 'TSNE'}:
        from sklearn.manifold import TSNE as Reducer  # noqa:N811
    elif model.upper() in {'LLE', 'LOCALLYLINEAREMBEDDING'}:
        from sklearn.manifold import LocallyLinearEmbedding as Reducer
    elif model.upper() == 'ISOMAP':
        from sklearn.manifold import Isomap as Reducer
    elif model.upper() in {'MDS', 'MULTIDIMENSIONALSCALING'}:
        from sklearn.manifold import MDS as Reducer  # noqa:N811
    elif model.upper() in {'SE', 'SPECTRAL', 'SPECTRALEMBEDDING'}:
        from sklearn.manifold import SpectralEmbedding as Reducer
    else:
        raise ValueError(f'invalid dimensionality reduction model: {model}')
    return Reducer, kwargs


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
    :param kwargs: Keyword arguments to be passed through to :func:`pipeline_from_path`.
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
    :param kwargs: Keyword arguments to be passed through to :func:`pipeline_from_config`.
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
    **kwargs,
) -> PipelineResult:
    """Run the pipeline with configuration in a JSON file at the given path.

    :param path: The path to an experiment JSON file
    """
    with open(path) as file:
        config = json.load(file)
    return pipeline_from_config(
        config=config,
        **kwargs,
    )


def pipeline_from_config(
    config: Mapping[str, Any],
    **kwargs,
) -> PipelineResult:
    """Run the pipeline with a configuration dictionary.

    :param config: The experiment configuration dictionary
    """
    metadata, pipeline_kwargs = config['metadata'], config['pipeline']
    title = metadata.get('title')
    if title is not None:
        logger.info(f'Running: {title}')

    return pipeline(
        metadata=metadata,
        **pipeline_kwargs,
        **kwargs,
    )


def pipeline(  # noqa: C901
    *,
    # 1. Dataset
    dataset: Union[None, str, DataSet, Type[DataSet]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training: Union[None, TriplesFactory, str] = None,
    testing: Union[None, TriplesFactory, str] = None,
    validation: Union[None, TriplesFactory, str] = None,
    evaluation_entity_whitelist: Optional[Collection[str]] = None,
    evaluation_relation_whitelist: Optional[Collection[str]] = None,
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
    # 9. Tracking
    result_tracker: Union[None, str, Type[ResultTracker]] = None,
    result_tracker_kwargs: Optional[Mapping[str, Any]] = None,
    # Misc
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
    :param training:
        A triples factory with training instances or path to the training file if a a dataset was not specified
    :param testing:
        A triples factory with training instances or path to the test file if a dataset was not specified
    :param validation:
        A triples factory with validation instances or path to the validation file if a dataset was not specified
    :param evaluation_entity_whitelist:
        Optional restriction of evaluation to triples containing *only* these entities. Useful if the downstream task
        is only interested in certain entities, but the relational patterns with other entities improve the entity
        embedding quality.
    :param evaluation_relation_whitelist:
        Optional restriction of evaluation to triples containing *only* these relations. Useful if the downstream task
        is only interested in certain relation, but the relational patterns with other relations improve the entity
        embedding quality.

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

    :param result_tracker:
        The ResultsTracker class or name
    :param result_tracker_kwargs:
        The keyword arguments passed to the results tracker on instantiation

    :param metadata:
        A JSON dictionary to store with the experiment
    :param use_testing_data:
        If true, use the testing triples. Otherwise, use the validation triples. Defaults to true - use testing triples.
    """
    if random_seed is None:
        random_seed = random_non_negative_int()
        logger.warning(f'No random seed is specified. Setting to {random_seed}.')
    set_random_seed(random_seed)

    result_tracker_cls: Type[ResultTracker] = get_result_tracker_cls(result_tracker)
    result_tracker = result_tracker_cls(**(result_tracker_kwargs or {}))

    if not metadata:
        metadata = {}
    title = metadata.get('title')

    # Start tracking
    result_tracker.start_run(run_name=title)

    device = resolve_device(device)

    dataset_instance: DataSet = get_dataset(
        dataset=dataset,
        dataset_kwargs=dataset_kwargs,
        training=training,
        testing=testing,
        validation=validation,
    )
    if dataset is not None:
        result_tracker.log_params(dict(dataset=dataset_instance.get_normalized_name()))
    else:  # means that dataset was defined by triples factories
        result_tracker.log_params(dict(dataset='<user defined>'))

    training, testing, validation = dataset_instance.training, dataset_instance.testing, dataset_instance.validation
    # evaluation restriction to a subset of entities/relations
    if any(f is not None for f in (evaluation_entity_whitelist, evaluation_relation_whitelist)):
        testing = testing.new_with_restriction(
            entities=evaluation_entity_whitelist,
            relations=evaluation_relation_whitelist,
        )
        if validation is not None:
            validation = validation.new_with_restriction(
                entities=evaluation_entity_whitelist,
                relations=evaluation_relation_whitelist,
            )

    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs.update(preferred_device=device)
    model_kwargs.setdefault('random_seed', random_seed)

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

    model = get_model_cls(model)
    model_instance: Model = model(
        triples_factory=training,
        **model_kwargs,
    )
    # Log model parameters
    result_tracker.log_params(params=dict(cls=model.__name__, kwargs=model_kwargs), prefix='model')

    optimizer = get_optimizer_cls(optimizer)
    training_loop = get_training_loop_cls(training_loop)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    # Log optimizer parameters
    result_tracker.log_params(params=dict(cls=optimizer.__name__, kwargs=optimizer_kwargs), prefix='optimizer')
    optimizer_instance = optimizer(
        params=model_instance.get_grad_params(),
        **optimizer_kwargs,
    )

    result_tracker.log_params(params=dict(cls=training_loop.__name__), prefix='training_loop')
    if negative_sampler is None:
        training_loop_instance: TrainingLoop = training_loop(
            model=model_instance,
            optimizer=optimizer_instance,
        )
    elif training_loop is not SLCWATrainingLoop:
        raise ValueError('Can not specify negative sampler with LCWA')
    else:
        negative_sampler = get_negative_sampler_cls(negative_sampler)
        result_tracker.log_params(
            params=dict(cls=negative_sampler.__name__, kwargs=negative_sampler_kwargs),
            prefix='negative_sampler',
        )
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
        evaluation_triples_factory=validation,
        result_tracker=result_tracker,
        **stopper_kwargs,
    )

    training_kwargs.setdefault('num_epochs', 5)
    training_kwargs.setdefault('batch_size', 256)
    result_tracker.log_params(params=training_kwargs, prefix='training')

    # Add logging for debugging
    logging.debug("Run Pipeline based on following config:")
    if dataset is not None:
        logging.debug(f"dataset: {dataset}")
        logging.debug(f"dataset_kwargs: {dataset_kwargs}")
    else:
        logging.debug('training: %s', training.path)
        logging.debug('testing: %s', testing.path)
        if validation:
            logging.debug('validation: %s', validation.path)
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
        mapped_triples = testing.mapped_triples
    else:
        mapped_triples = validation.mapped_triples

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
