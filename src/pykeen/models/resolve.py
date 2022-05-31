# -*- coding: utf-8 -*-

"""A :class:`pykeen.models.ERModel` can be constructed from :class:`pykeen.nn.modules.Interaction`.

The new style-class, :class:`pykeen.models.ERModel` abstracts the interaction away from the representations
such that different interactions can be used interchangably. A new model can be constructed directly from the
interaction module, given a ``dimensions`` mapping. In each :class:`pykeen.nn.modules.Interaction`, there
is a field called ``entity_shape`` and ``relation_shape`` that allows for using eigen-notation for defining
the different dimensions of the model. Most models share the ``d`` dimensionality for both the entity and relation
vectors. Some (but not all) exceptions are:

- :class:`pykeen.nn.modules.RESCALInteraction`, which uses a square matrix for relations written as ``dd``
- :class:`pykeen.nn.modules.TransDInteraction`, which uses ``d`` for entity shape and ``e`` for a different
  relation shape.

With this in mind, you'll have to investigate the dimensions of the vectors through the PyKEEN documentation.
If you're implementing your own, you have control over this and will know which dimensions to specify (though
the ``d`` for both entities and relations is standard). As a shorthand for ``{'d': value}``, you can directly
pass ``value`` for the dimension and it will be automatically interpreted as the ``{'d': value}``.

Make a model class from lookup of an interaction module class:

>>> from pykeen.nn.modules import TransEInteraction
>>> from pykeen.models import make_model_cls
>>> embedding_dim = 3
>>> model_cls = make_model_cls(
...     dimensions={"d": embedding_dim},
...     interaction='TransE',
...     interaction_kwargs={'p': 2},
... )

If there's only one dimension in the ``entity_shapes`` and ``relation_shapes``, it can
be directly given as an integer as a shortcut.

>>> # Implicitly can also be written as:
>>> model_cls_alt = make_model_cls(
...     dimensions=embedding_dim,
...     interaciton='TransE',
...     interaction_kwargs={'p': 2},
... )

Make a model class from an interaction module class:

>>> from pykeen.nn.modules import TransEInteraction
>>> from pykeen.models import make_model_cls
>>> embedding_dim = 3
>>> model_cls = make_model_cls({"d": embedding_dim}, TransEInteraction, {'p': 2})

Make a model class from an instantiated interaction module:

>>> from pykeen.nn.modules import TransEInteraction
>>> from pykeen.models import make_model_cls
>>> embedding_dim = 3
>>> model_cls = make_model_cls({"d": embedding_dim}, TransEInteraction(p=2))

All of these model classes can be passed directly into the ``model``
argument of :func:`pykeen.pipeline.pipeline`.
"""

import logging
from typing import Any, Mapping, Optional, Sequence, Tuple, Type, Union

from class_resolver import OptionalKwargs

from .nbase import ERModel
from ..nn.modules import Interaction, interaction_resolver
from ..typing import HeadRepresentation, RelationRepresentation, TailRepresentation

__all__ = [
    "make_model",
    "make_model_cls",
]

logger = logging.getLogger(__name__)


def make_model(
    dimensions: Union[int, Mapping[str, int]],
    interaction: Union[
        str,
        Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation],
        Type[Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation]],
    ],
    interaction_kwargs: Optional[Mapping[str, Any]] = None,
    entity_representations_kwargs: OptionalKwargs = None,
    relation_representations_kwargs: OptionalKwargs = None,
    **kwargs,
) -> ERModel:
    """Build a model from an interaction class hint (name or class)."""
    model_cls = make_model_cls(
        dimensions=dimensions,
        interaction=interaction,
        interaction_kwargs=interaction_kwargs,
        entity_representations_kwargs=entity_representations_kwargs,
        relation_representations_kwargs=relation_representations_kwargs,
    )
    return model_cls(**kwargs)


class DimensionError(ValueError):
    """Raised when the wrong dimensions were supplied."""

    def __init__(self, given, expected):
        """
        Initialize the error.

        :param given:
            the given keys
        :param expected:
            the expected keys
        """
        self.given = given
        self.expected = expected

    def __str__(self):
        return f"Expected dimensions dictionary with keys {self.expected} but got keys {self.given}"


def make_model_cls(
    dimensions: Union[int, Mapping[str, int]],
    interaction: Union[
        str,
        Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation],
        Type[Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation]],
    ],
    interaction_kwargs: Optional[Mapping[str, Any]] = None,
    entity_representations_kwargs: OptionalKwargs = None,
    relation_representations_kwargs: OptionalKwargs = None,
) -> Type[ERModel]:
    """Build a model class from an interaction class hint (name or class)."""
    if isinstance(interaction, Interaction):
        interaction_instance = interaction
    else:
        interaction_instance = interaction_resolver.make(interaction, interaction_kwargs)

    entity_representations_kwargs, relation_representations_kwargs = _normalize_representation_kwargs(
        dimensions=dimensions,
        interaction=interaction_instance.__class__,  # type: ignore
        entity_representations_kwargs=entity_representations_kwargs,
        relation_representations_kwargs=relation_representations_kwargs,
    )

    # TODO pack/unpack dimensions as default kwargs such that they don't actually need to be used
    #  to create the class

    class ChildERModel(ERModel[HeadRepresentation, RelationRepresentation, TailRepresentation]):
        """A ad-hoc child class of ERModel."""

        def __init__(self, **kwargs) -> None:
            """Initialize the model."""
            super().__init__(
                interaction=interaction_instance,
                entity_representations_kwargs=entity_representations_kwargs,
                relation_representations_kwargs=relation_representations_kwargs,
                **kwargs,
            )

    ChildERModel._interaction = interaction_instance

    return ChildERModel


def _normalize_representation_kwargs(
    dimensions: Union[int, Mapping[str, int]],
    interaction: Type[Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation]],
    entity_representations_kwargs: OptionalKwargs,
    relation_representations_kwargs: OptionalKwargs,
) -> Tuple[Sequence[OptionalKwargs], Sequence[OptionalKwargs]]:
    # TODO: update to hint + kwargs
    if isinstance(dimensions, int):
        dimensions = {"d": dimensions}
    assert isinstance(dimensions, dict)
    if set(dimensions) < interaction.get_dimensions():
        raise DimensionError(set(dimensions), interaction.get_dimensions())
    if entity_representations_kwargs is None:
        # TODO: Does not work for interactions with separate tail_entity_shape (i.e., ConvE)
        if interaction._tail_entity_shape is not None:
            raise NotImplementedError
        entity_representations_kwargs = [
            dict(shape=tuple(dimensions[d] for d in shape)) for shape in interaction.entity_shape
        ]
    elif not isinstance(entity_representations_kwargs, Sequence):
        entity_representations_kwargs = [entity_representations_kwargs]
    if relation_representations_kwargs is None:
        relation_representations_kwargs = [
            dict(shape=tuple(dimensions[d] for d in shape)) for shape in interaction.relation_shape
        ]
    elif not isinstance(relation_representations_kwargs, Sequence):
        relation_representations_kwargs = [relation_representations_kwargs]
    return entity_representations_kwargs, relation_representations_kwargs
