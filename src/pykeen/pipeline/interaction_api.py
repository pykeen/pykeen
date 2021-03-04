# -*- coding: utf-8 -*-

"""The interaction API is a pipeline cenetered around interaction modules instead of models."""

from typing import Any, Mapping, Optional, Type, Union

from .api import PipelineResult, pipeline
from ..models import make_model_cls
from ..models.nbase import EmbeddingSpecificationHint
from ..nn.modules import Interaction

__all__ = [
    'interaction_pipeline',
]


def interaction_pipeline(
    dimensions: Union[int, Mapping[str, int]],
    interaction: Union[str, Interaction, Type[Interaction]],
    interaction_kwargs: Optional[Mapping[str, Any]] = None,
    entity_representations: EmbeddingSpecificationHint = None,
    relation_representations: EmbeddingSpecificationHint = None,
    **kwargs,
) -> PipelineResult:
    """Run the pipeline on a custom interaction model.

    :param dimensions: The dimensions dictionary for :func:`make_model_cls`
    :param interaction: The interaction class name, class, or instance
    :param interaction_kwargs: The keyword arguments to be used when instantiating the interaction class
        if a string or class is passed.
    :param entity_representations: The entity representations, embedding specifications,
        or (default) none if they should be inferred.
    :param relation_representations: The relation representations, embedding specifications,
        or (default) none if they should be inferred.
    :param kwargs: Remaining arguments passed to :func:`pipeline`.
    :return: A pipeline result package

    :raises ValueError: if the keyword arguments includes ``model`` - these should not be
        specified at the same time. If you have a model, use the :func:`pipeline` function.
    """
    if 'model' in kwargs:
        raise ValueError('should not include both model and interaction')
    model_cls = make_model_cls(
        interaction=interaction,
        dimensions=dimensions,
        interaction_kwargs=interaction_kwargs,
        entity_representations=entity_representations,
        relation_representations=relation_representations,
    )
    return pipeline(model=model_cls, **kwargs)
