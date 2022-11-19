# -*- coding: utf-8 -*-

"""Implementation of DistMA."""

from typing import Any, ClassVar, Mapping, Optional

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.modules import DistMAInteraction
from ...typing import Hint, Initializer, Normalizer

__all__ = [
    "DistMA",
]


class DistMA(ERModel):
    r"""An implementation of DistMA from [shi2019]_.

    ---
    citation:
        author: Shi
        year: 2019
        link: https://www.aclweb.org/anthology/D19-1075.pdf
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        embedding_dim: int = 256,
        entity_initializer: Hint[Initializer] = None,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_normalizer: Hint[Normalizer] = None,
        entity_normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = None,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize DistMA via the :class:`pykeen.nn.modules.DistMAInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$.
        :param entity_initializer: Entity initializer function. Defaults to None
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer
        :param entity_normalizer: Entity normalizer function. Defaults to None
        :param entity_normalizer_kwargs: Keyword arguments to be used when calling the entity normalizer
        :param relation_initializer: Relation initializer function. Defaults to None
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=DistMAInteraction,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                initializer_kwargs=entity_initializer_kwargs,
                normalizer=entity_normalizer,
                normalizer_kwargs=entity_normalizer_kwargs,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                initializer_kwargs=relation_initializer_kwargs,
            ),
            **kwargs,
        )
