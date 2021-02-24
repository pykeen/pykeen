# -*- coding: utf-8 -*-

"""Implementation of PairRE."""

from typing import Any, ClassVar, Mapping

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import EmbeddingSpecification
from ...nn.init import xavier_normal_
from ...nn.modules import PairREInteraction
from ...typing import Hint, Initializer

__all__ = [
    'PairRE',
]


class PairRE(ERModel):
    r"""An implementation of PairRE from [chao2020]_.

    ---
    citation:
        author: Chao
        year: 2020
        link: http://arxiv.org/abs/2011.03798
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        p=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        embedding_dim: int = 200,
        p: int = 2,
        power_norm: bool = True,
        entity_initializer: Hint[Initializer] = xavier_normal_,
        relation_initializer: Hint[Initializer] = xavier_normal_,
        **kwargs,
    ) -> None:
        r"""Initialize PairRE via the :class:`pykeen.nn.modules.PairREInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$. Defaults to 200. Is usually $d \in [50, 300]$.
        :param p: The $l_p$ norm. Defaults to 2.
        :param power_norm: Should the power norm be used? Defaults to true.

        .. warning:: Due to the lack of an official implementations, not all details are known.
        """
        super().__init__(
            interaction=PairREInteraction(p=p, power_norm=power_norm),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                ),
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                ),
            ],
            **kwargs,
        )
