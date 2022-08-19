# -*- coding: utf-8 -*-

"""Plotting utilities for the pipeline results."""

import logging
from typing import Callable, Mapping, Optional, Set

from ..losses import loss_resolver
from ..models.nbase import ERModel
from ..nn.representation import Representation
from ..stoppers import EarlyStopper

__all__ = [
    "plot_losses",
    "plot_early_stopping",
    "plot_er",
    "plot",
]

logger = logging.getLogger(__name__)

REDUCER_RELATION_WHITELIST = {"PCA"}


def plot_losses(pipeline_result, *, ax=None):
    """Plot the losses per epoch."""
    import seaborn as sns

    sns.set_style("darkgrid")

    ax = _ensure_ax(ax)
    rv = sns.lineplot(x=range(len(pipeline_result.losses)), y=pipeline_result.losses, ax=ax)

    loss_name = loss_resolver.normalize_inst(pipeline_result.model.loss)
    ax.set_ylabel(f"{loss_name} Loss")
    ax.set_xlabel("Epoch")
    ax.set_title(pipeline_result.title if pipeline_result.title is not None else "Losses Plot")
    return rv


def plot_early_stopping(pipeline_result, *, ax=None, lineplot_kwargs=None):
    """Plot the evaluations during early stopping."""
    import seaborn as sns

    if not isinstance(pipeline_result.stopper, EarlyStopper) or not pipeline_result.stopper.results:
        raise ValueError

    ax = _ensure_ax(ax)

    x = [(1 + e) * pipeline_result.stopper.frequency for e in range(len(pipeline_result.stopper.results))]
    rv = sns.lineplot(x=x, y=pipeline_result.stopper.results, ax=ax, marker="o", **(lineplot_kwargs or {}))

    ax.set_ylabel(pipeline_result.stopper.metric)
    ax.set_xlabel("Epoch")
    ax.set_title(pipeline_result.title if pipeline_result.title is not None else "Early Stopper Evaluation Plot")
    return rv


def build_representation_getter(relation: bool = False, index: int = 0) -> Callable[[ERModel], Representation]:
    """
    Build a representation getter.

    :param relation:
        whether to get relation representations, or entity representations.
    :param index:
        the index of the representation to get

    :return:
        a function to get the representation.
    """

    def getter(model: ERModel) -> Representation:
        """
        Get a specific representation from model.

        :param model:
            the model

        :return:
            the representation
        """
        # cf. also https://github.com/pykeen/pykeen/issues/1071
        representations = model.relation_representations if relation else model.entity_representations
        return representations[index]

    return getter


def plot_er(  # noqa: C901
    pipeline_result,
    *,
    model: Optional[str] = None,
    entities: Optional[Set[str]] = None,
    relations: Optional[Set[str]] = None,
    apply_limits: bool = True,
    margin: float = 0.4,
    plot_entities: bool = True,
    plot_relations: Optional[bool] = None,
    annotation_x_offset: float = 0.02,
    annotation_y_offset: float = 0.03,
    entity_embedding_getter=None,
    relation_embedding_getter=None,
    ax=None,
    subtitle: Optional[str] = None,
    **kwargs,
):
    """Plot the reduced entities and relation vectors in 2D.

    :param pipeline_result: The result returned by :func:`pykeen.pipeline.pipeline`.
    :param model: The dimensionality reduction model from :mod:`sklearn`. Defaults to PCA.
        Can also use KPCA, GRP, SRP, TSNE, LLE, ISOMAP, MDS, or SE.
    :param entities: A subset of entities to plot
    :param relations: A subset of relations to plot
    :param apply_limits: Should the x and y limits be applied?
    :param margin: The margin size around the minimum/maximum x and y values
    :param plot_entities: If true, plot the entities based on their reduced embeddings
    :param plot_relations: By default, this is only enabled on translational distance models
        like :class:`pykeen.models.TransE`.
    :param annotation_x_offset: X offset of label from entity position
    :param annotation_y_offset: Y offset of label from entity position
    :param entity_embedding_getter: A function that takes a model and returns its entity embeddings. If none,
        defaults to :func:`_default_entity_embedding_getter`, which just gets ``model.entity_embeddings``. Note,
        the default only works with old-style PyKEEN models.
    :param relation_embedding_getter: A function that takes a model and returns its relation embeddings. If none,
        defaults to :func:`_default_relation_embedding_getter`, which just gets ``model.relation_embeddings``. Note,
        the default only works with old-style PyKEEN models.
    :param ax: The matplotlib axis, if pre-defined
    :param subtitle: A user-defined subtitle. Is inferred if not given. Pass an empty string to not use a subtitle.
    :param kwargs: The keyword arguments passed to `__init__()` of
        the reducer class (e.g., PCA, TSNE)
    :returns: The axis

    :raises ValueError: if entity plotting and relation plotting are both turned off

    .. warning::

        Plotting relations and entities on the same plot is only
        meaningful for translational distance models like TransE.
    """
    import seaborn as sns

    if not plot_entities and not plot_relations:
        raise ValueError

    if plot_relations is None:  # automatically set to true for translational models, false otherwise
        plot_relations = pipeline_result.model.__class__.__name__.lower().startswith("trans")

    if model is None:
        model = "PCA"
    reducer_cls, reducer_kwargs = _get_reducer_cls(model, **kwargs)
    if plot_relations and reducer_cls.__name__ not in REDUCER_RELATION_WHITELIST:
        raise ValueError(f"Can not use reducer {reducer_cls} when projecting relations. Will result in nonsense")
    reducer = reducer_cls(n_components=2, **reducer_kwargs)

    ax = _ensure_ax(ax)

    sns.set_style("whitegrid")

    if entity_embedding_getter is None:
        entity_embedding_getter = build_representation_getter(relation=False, index=0)
    if relation_embedding_getter is None:
        relation_embedding_getter = build_representation_getter(relation=True, index=0)

    if plot_relations and plot_entities:
        e_embeddings, e_reduced = _reduce_embeddings(entity_embedding_getter(pipeline_result.model), reducer, fit=True)
        r_embeddings, r_reduced = _reduce_embeddings(
            relation_embedding_getter(pipeline_result.model),
            reducer,
            fit=False,
        )

        xmax = max(r_embeddings[:, 0].max(), e_embeddings[:, 0].max()) + margin
        xmin = min(r_embeddings[:, 0].min(), e_embeddings[:, 0].min()) - margin
        ymax = max(r_embeddings[:, 1].max(), e_embeddings[:, 1].max()) + margin
        ymin = min(r_embeddings[:, 1].min(), e_embeddings[:, 1].min()) - margin
    elif plot_relations:
        e_embeddings, e_reduced = None, False
        r_embeddings, r_reduced = _reduce_embeddings(
            relation_embedding_getter(pipeline_result.model),
            reducer,
            fit=True,
        )

        xmax = r_embeddings[:, 0].max() + margin
        xmin = r_embeddings[:, 0].min() - margin
        ymax = r_embeddings[:, 1].max() + margin
        ymin = r_embeddings[:, 1].min() - margin
    elif plot_entities:
        e_embeddings, e_reduced = _reduce_embeddings(entity_embedding_getter(pipeline_result.model), reducer, fit=True)
        r_embeddings, r_reduced = None, False

        xmax = e_embeddings[:, 0].max() + margin
        xmin = e_embeddings[:, 0].min() - margin
        ymax = e_embeddings[:, 1].max() + margin
        ymin = e_embeddings[:, 1].min() - margin
    else:
        raise ValueError  # not even possible

    if subtitle is not None:
        pass  # a specific subtitle has been given
    elif not e_reduced and not r_reduced:
        subtitle = ""
    elif reducer_kwargs:
        _subtitle_ending = ", ".join(f"{key}={value}" for key, value in reducer_kwargs.items())
        subtitle = f" using {reducer_cls.__name__} ({_subtitle_ending})"
    else:
        subtitle = f" using {reducer_cls.__name__}"

    if plot_entities:
        entity_id_to_label = pipeline_result.training.entity_id_to_label
        for entity_id, entity_reduced_embedding in enumerate(e_embeddings):
            entity_label = entity_id_to_label[entity_id]
            if entities and entity_label not in entities:
                continue
            x, y = entity_reduced_embedding
            ax.scatter(x, y, color="black")
            ax.annotate(entity_label, (x + annotation_x_offset, y + annotation_y_offset))

    if plot_relations:
        relation_id_to_label = pipeline_result.training.relation_id_to_label
        for relation_id, relation_reduced_embedding in enumerate(r_embeddings):
            relation_label = relation_id_to_label[relation_id]
            if relations and relation_label not in relations:
                continue
            x, y = relation_reduced_embedding
            ax.arrow(0, 0, x, y, color="black")
            ax.annotate(relation_label, (x + annotation_x_offset, y + annotation_y_offset))

    if plot_entities and plot_relations:
        ax.set_title(f"Entity/Relation Plot{subtitle}")
    elif plot_entities:
        ax.set_title(f"Entity Plot{subtitle}")
    elif plot_relations:
        ax.set_title(f"Relation Plot{subtitle}")

    if apply_limits:
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    return ax


def _ensure_ax(ax):
    if ax is not None:
        return ax

    import matplotlib.pyplot as plt

    return plt.gca()


def _reduce_embeddings(embedding: Representation, reducer, fit: bool = False):
    embeddings_numpy = embedding(indices=None).detach().cpu().numpy()
    if embeddings_numpy.shape[1] == 2:
        logger.debug("not reducing entity embeddings, already dim=2")
        return embeddings_numpy, False
    elif fit:
        return reducer.fit_transform(embeddings_numpy), True
    else:
        return reducer.transform(embeddings_numpy), True


def _get_reducer_cls(model: str, **kwargs):
    """Get the model class by name and default kwargs.

    :param model: The name of the model. Can choose from: PCA, KPCA, GRP,
        SRP, TSNE, LLE, ISOMAP, MDS, or SE.
    :param kwargs: Keyword arguments that will get passed through and modified based on the chosen model.
    :return: A pair of a reducer class from :mod:`sklearn` and the modified kwargs.

    :raises ValueError: if invalid model name is passed
    """
    # TODO: use a class-resolver?
    if model.upper() == "PCA":
        from sklearn.decomposition import PCA as Reducer  # noqa:N811
    elif model.upper() == "KPCA":
        kwargs.setdefault("kernel", "rbf")
        from sklearn.decomposition import KernelPCA as Reducer
    elif model.upper() == "GRP":
        from sklearn.random_projection import GaussianRandomProjection as Reducer
    elif model.upper() == "SRP":
        from sklearn.random_projection import SparseRandomProjection as Reducer
    elif model.upper() in {"T-SNE", "TSNE"}:
        from sklearn.manifold import TSNE as Reducer  # noqa:N811
    elif model.upper() in {"LLE", "LOCALLYLINEAREMBEDDING"}:
        from sklearn.manifold import LocallyLinearEmbedding as Reducer
    elif model.upper() == "ISOMAP":
        from sklearn.manifold import Isomap as Reducer
    elif model.upper() in {"MDS", "MULTIDIMENSIONALSCALING"}:
        from sklearn.manifold import MDS as Reducer  # noqa:N811
    elif model.upper() in {"SE", "SPECTRAL", "SPECTRALEMBEDDING"}:
        from sklearn.manifold import SpectralEmbedding as Reducer
    else:
        raise ValueError(f"invalid dimensionality reduction model: {model}")
    return Reducer, kwargs


def plot(pipeline_result, er_kwargs: Optional[Mapping[str, str]] = None, figsize=(10, 4)):
    """Plot all plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    pipeline_result.plot_losses(ax=axes[0])
    pipeline_result.plot_er(ax=axes[1], **(er_kwargs or {}))

    plt.tight_layout()
    return fig, axes
