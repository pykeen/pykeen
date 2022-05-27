# -*- coding: utf-8 -*-

"""Implementation of UM."""

from typing import Any, ClassVar, Mapping

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.init import xavier_normal_
from ...nn.modules import UMInteraction
from ...typing import Hint, Initializer

__all__ = [
    "UM",
]


class UM(ERModel):
    r"""An implementation of the Unstructured Model (UM) published by [bordes2014]_.

    UM computes the distance between head and tail entities then applies the $l_p$ norm.

    .. math::

        f(h, r, t) = - \|\textbf{e}_h  - \textbf{e}_t\|_p^2

    A small distance between the embeddings for the head and tail entity indicates a plausible triple. It is
    appropriate for networks with a single relationship type that is undirected.

    .. warning::

        In UM, neither the relations nor the directionality are considered, so it can't distinguish between them.
        However, it may serve as a baseline for comparison against relation-aware models.
    ---
    name: Unstructured Model
    citation:
        author: Bordes
        year: 2014
        link: https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 1,
        entity_initializer: Hint[Initializer] = xavier_normal_,
        **kwargs,
    ) -> None:
        r"""Initialize UM.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The $l_p$ norm. Usually 1 for UM.
        :param entity_initializer: The initializer for the entity embeddings.
            Defaults to the xavier normal distribution.
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=UMInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations=[],
            **kwargs,
        )
