# -*- coding: utf-8 -*-

"""Implementation of TorusE."""

from typing import Any, ClassVar, Mapping, Optional

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.modules import TorusEInteraction
from ...typing import Hint, Initializer, Normalizer

__all__ = [
    "TorusE",
]


class TorusE(ERModel):
    r"""An implementation of TorusE from [ebisu2018]_.

    ---
    citation:
        author: Ebisu
        year: 2018
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16227
        arxiv: 1711.05435
        github: TakumaE/TorusE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        p=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        embedding_dim: int = 256,
        p: int = 2,
        power_norm: bool = False,
        entity_initializer: Hint[Initializer] = None,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_normalizer: Hint[Normalizer] = None,
        entity_normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = None,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize TorusE via the :class:`pykeen.nn.modules.TorusEInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$.
        :param p: The p for the norm.
        :param power_norm: Whether to use the p-th power of the L_p norm instead.
        :param entity_initializer: Entity initializer function. Defaults to None
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer
        :param entity_normalizer: Entity normalizer function. Defaults to None
        :param entity_normalizer_kwargs: Keyword arguments to be used when calling the entity normalizer
        :param relation_initializer: Relation initializer function. Defaults to None
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=TorusEInteraction,
            interaction_kwargs=dict(p=p, power_norm=power_norm),
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
