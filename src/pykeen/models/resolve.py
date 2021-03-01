# -*- coding: utf-8 -*-

"""Generation of ad-hoc model classes."""

import logging
from typing import Any, Mapping, Optional, Sequence, Tuple, Type, Union

from .nbase import ERModel, EmbeddingSpecificationHint
from ..nn import EmbeddingSpecification, RepresentationModule
from ..nn.modules import Interaction, interaction_resolver
from ..typing import Hint

__all__ = [
    'model_builder',
    'model_from_interaction',
]

logger = logging.getLogger(__name__)


def _normalize_entity_representations(
    dimensions: Mapping[str, int],
    interaction: Type[Interaction],
    entity_representations: EmbeddingSpecificationHint,
    relation_representations: EmbeddingSpecificationHint,
) -> Tuple[Sequence, Sequence]:
    if entity_representations is None:
        # TODO: Does not work for interactions with separate tail_entity_shape (i.e., ConvE)
        if interaction.tail_entity_shape is not None:
            raise NotImplementedError
        entity_representations = [
            EmbeddingSpecification(shape=tuple(
                dimensions[d]
                for d in shape
            ))
            for shape in interaction.entity_shape
        ]
    if relation_representations is None:
        relation_representations = [
            EmbeddingSpecification(shape=tuple(
                dimensions[d]
                for d in shape
            ))
            for shape in interaction.relation_shape
        ]
    return entity_representations, relation_representations


def model_instance_builder(
    dimensions: Mapping[str, Any],
    interaction: Hint[Interaction] = None,
    interaction_kwargs: Optional[Mapping[str, Any]] = None,
    entity_representations: EmbeddingSpecificationHint = None,
    relation_representations: EmbeddingSpecificationHint = None,
    **kwargs,
) -> ERModel:
    """Build a model from an interaction class hint (name or class)."""
    interaction_instance = interaction_resolver.make(interaction, interaction_kwargs)
    entity_representations, relation_representations = _normalize_entity_representations(
        dimensions=dimensions,
        interaction=interaction,
        entity_representations=entity_representations,
        relation_representations=relation_representations,
    )
    return ERModel(
        interaction=interaction_instance,
        entity_representations=entity_representations,
        relation_representations=relation_representations,
        **kwargs,
    )


def model_builder(
    interaction: Hint[Interaction] = None,
    interaction_kwargs: Optional[Mapping[str, Any]] = None,
    entity_representations: EmbeddingSpecificationHint = None,
    relation_representations: EmbeddingSpecificationHint = None,
) -> Type[ERModel]:
    """Build a model class from an interaction class hint (name or class)."""
    interaction_instance = interaction_resolver.make(interaction, interaction_kwargs)
    return model_from_interaction(
        interaction=interaction_instance,
        entity_representations=entity_representations,
        relation_representations=relation_representations,
    )


def model_from_interaction(
    interaction: Interaction,
    entity_representations: EmbeddingSpecificationHint = None,
    relation_representations: EmbeddingSpecificationHint = None,
) -> Type[ERModel]:
    """Build a model class from an interaction class instance."""

    class ChildERModel(ERModel):
        def __init__(self, **kwargs) -> None:
            """Initialize the model."""
            super().__init__(
                interaction=interaction,
                entity_representations=entity_representations,
                relation_representations=relation_representations,
                **kwargs,
            )

    ChildERModel._interaction = interaction

    return ChildERModel


# FIXME this function is borrowed from PR #107 and originally implemented by @mberr.
#  It needs a bit of work
def _resolve_representations(
    num_representations: int,
    shape: Sequence[str],
    representations: Sequence[Union[None, RepresentationModule, EmbeddingSpecification]],
    dimensions: Mapping[str, int],
) -> Sequence[RepresentationModule]:
    """
    Resolve representations.

    :param num_representations:
        The number of representations, e.g., number of entities.
    :param shape: length: num_representations
        The symbolic shapes for each individual representation. Each shape is a sequence of characters. Each character
         is resolved to an integer dimension, either by looking it up in the dimensions dictionary, or by sharing the
         symbolic shape with any of the other representations with given shape.
    :param representations:
        The representation specifications. These are either `EmbeddingSpecification`'s, or instantiated representations.
        Values of `None` correspond to default `EmbeddingSpecification`s.
    :param dimensions:
        A mapping from symbolic shapes to actual dimensions.

    :return:
        A sequence of representation instances of size `num_representations`.
    """
    if not isinstance(representations, Sequence):
        if isinstance(representations, RepresentationModule):
            # FIXME: Error message; also, why is this a problem.
            raise ValueError
        # share same embedding specification for all representations
        # TODO: why? they should usually have different ones
        representations = [representations] * len(shape)

    # shallow copy to avoid side-effects
    dimensions = dict(**dimensions)

    # TODO: Start with instantiated representations, then embedding specification and then None
    result = []
    for symbolic_shape, representation in zip(shape, representations):
        if representation is None:
            representation = EmbeddingSpecification()
        if isinstance(representation, RepresentationModule):
            if representation.max_id < num_representations:
                # FIXME: Error message
                raise ValueError
            elif representation.max_id > num_representations:
                logger.warning(
                    f"{representation} does provide representations for more than the requested {num_representations}."
                    f"While this is not necessarily an error, be aware that these representations will not be trained.",
                )
            for name, size in zip(symbolic_shape, representation.shape):
                expected_dimension = dimensions.get(name)
                if expected_dimension is not None and expected_dimension != size:
                    # FIXME: Error message
                    raise ValueError
                dimensions[name] = size
        elif isinstance(representation, EmbeddingSpecification):
            # set actual shape
            actual_shape = tuple([dimensions[name] for name in symbolic_shape])
            representation.shape = actual_shape

            # create embedding
            representation = representation.make(num_embeddings=num_representations)
        else:
            raise AssertionError
        result.append(representation)

    return result
