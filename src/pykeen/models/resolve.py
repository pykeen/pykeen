# -*- coding: utf-8 -*-

"""Generation of ad-hoc model classes."""

import logging
from typing import Any, Mapping, Optional, Sequence, Type, Union

from .nbase import ERModel, EmbeddingSpecificationHint
from ..nn import EmbeddingSpecification, RepresentationModule
from ..nn.modules import Interaction, interaction_resolver
from ..typing import Hint, OneOrSequence

__all__ = [
    'model_builder',
    'model_from_interaction',
]

logger = logging.getLogger(__name__)


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
    representations: OneOrSequence[Union[None, RepresentationModule, EmbeddingSpecification]],
    dimensions: Mapping[str, int],
) -> Sequence[RepresentationModule]:
    if not isinstance(representations, Sequence):
        if isinstance(representations, RepresentationModule):
            raise ValueError
        representations = [representations] * len(shape)

    dimensions = dict(**dimensions)
    result = []
    for symbolic_shape, representation in zip(shape, representations):
        if representation is None:
            representation = EmbeddingSpecification()
        if isinstance(representation, RepresentationModule):
            if representation.max_id < num_representations:
                raise ValueError
            elif representation.max_id > num_representations:
                logger.warning(
                    f"{representation} does provide representations for more than the requested {num_representations}."
                    f"While this is not necessarily an error, be aware that these representations will not be trained.",
                )
            for name, size in zip(symbolic_shape, representation.shape):
                expected_dimension = dimensions.get(name)
                if expected_dimension is not None and expected_dimension != size:
                    raise ValueError
                dimensions[name] = size
        elif isinstance(representation, EmbeddingSpecification):
            actual_shape = tuple([dimensions[name] for name in symbolic_shape])
            representation = representation.make(
                num_embeddings=num_representations,
                embedding_dim=None,
                shape=actual_shape,
            )
        else:
            raise AssertionError
        result.append(representation)

    return result
