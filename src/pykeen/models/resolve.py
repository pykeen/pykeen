# -*- coding: utf-8 -*-

"""Generation of ad-hoc model classes."""

import logging
from typing import Any, Mapping, Optional, Sequence, Tuple, Type, Union

from .nbase import ERModel, EmbeddingSpecificationHint
from ..nn import EmbeddingSpecification, RepresentationModule
from ..nn.modules import Interaction, interaction_resolver
from ..typing import Hint

__all__ = [
    'model_instance_builder',
    'model_builder',
    'model_from_interaction',
]

logger = logging.getLogger(__name__)


def model_instance_builder(
    dimensions: Mapping[str, Any],
    interaction: Hint[Interaction] = None,
    interaction_kwargs: Optional[Mapping[str, Any]] = None,
    entity_representations: EmbeddingSpecificationHint = None,
    relation_representations: EmbeddingSpecificationHint = None,
    **kwargs,
) -> ERModel:
    """Build a model from an interaction class hint (name or class)."""
    model_cls = model_builder(
        dimensions=dimensions,
        interaction=interaction,
        interaction_kwargs=interaction_kwargs,
        entity_representations=entity_representations,
        relation_representations=relation_representations,
    )
    return model_cls(**kwargs)


def model_builder(
    dimensions: Mapping[str, Any],
    interaction: Hint[Interaction] = None,
    interaction_kwargs: Optional[Mapping[str, Any]] = None,
    entity_representations: EmbeddingSpecificationHint = None,
    relation_representations: EmbeddingSpecificationHint = None,
) -> Type[ERModel]:
    """Build a model class from an interaction class hint (name or class)."""
    interaction_instance = interaction_resolver.make(interaction, interaction_kwargs)
    return model_from_interaction(
        dimensions=dimensions,
        interaction=interaction_instance,
        entity_representations=entity_representations,
        relation_representations=relation_representations,
    )


def model_from_interaction(
    dimensions: Mapping[str, Any],
    interaction: Interaction,
    entity_representations: EmbeddingSpecificationHint = None,
    relation_representations: EmbeddingSpecificationHint = None,
) -> Type[ERModel]:
    """Build a model class from an interaction class instance."""
    entity_representations, relation_representations = _normalize_entity_representations(
        dimensions=dimensions,
        interaction=interaction.__class__,
        entity_representations=entity_representations,
        relation_representations=relation_representations,
    )

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


def _normalize_entity_representations(
    dimensions: Mapping[str, int],
    interaction: Type[Interaction],
    entity_representations: EmbeddingSpecificationHint,
    relation_representations: EmbeddingSpecificationHint,
) -> Tuple[
    Sequence[Union[EmbeddingSpecification, RepresentationModule]],
    Sequence[Union[EmbeddingSpecification, RepresentationModule]],
]:
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
    elif not isinstance(entity_representations, Sequence):
        entity_representations = [entity_representations]
    if relation_representations is None:
        relation_representations = [
            EmbeddingSpecification(shape=tuple(
                dimensions[d]
                for d in shape
            ))
            for shape in interaction.relation_shape
        ]
    elif not isinstance(relation_representations, Sequence):
        relation_representations = [relation_representations]
    return entity_representations, relation_representations
