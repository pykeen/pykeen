# -*- coding: utf-8 -*-

"""Implementation of MuRE."""
import functools
from typing import Any, ClassVar, Mapping

import torch

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import EmbeddingSpecification
from ...nn.init import xavier_normal_
from ...nn.modules import MuREInteraction
from ...typing import Hint, Initializer
from ...utils import compose

__all__ = [
    'MuRE',
]


class MuRE(ERModel):
    r"""An implementation of MuRE from [balazevic2019b]_.

    ---
    citation:
        author: Balažević
        year: 2019
        link: https://arxiv.org/abs/1905.09791
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        p=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        p: int = 2,
        power_norm: bool = True,
        entity_initializer: Hint[Initializer] = xavier_normal_,
        relation_initializer: Hint[Initializer] = xavier_normal_,
        **kwargs,
    ) -> None:
        r"""Initialize MuRE via the :class:`pykeen.nn.modules.MuREInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$. Defaults to 200. Is usually $d \in [50, 300]$.
        :param p: The $l_p$ norm. Defaults to 2.
        :param power_norm: Should the power norm be used? Defaults to true.
        """
        # comment:
        # https://github.com/ibalazevic/multirelational-poincare/blob/34523a61ca7867591fd645bfb0c0807246c08660/model.py#L52
        # uses float64
        super().__init__(
            interaction=MuREInteraction(p=p, power_norm=power_norm),
            entity_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=compose(
                        torch.nn.init.normal_,
                        lambda x: 1.0e-03 * x,
                    ),
                ),
                # entity bias for head
                EmbeddingSpecification(
                    shape=tuple(),  # scalar
                    initializer=torch.nn.init.zeros_,
                ),
                # entity bias for tail
                EmbeddingSpecification(
                    shape=tuple(),  # scalar
                    initializer=torch.nn.init.zeros_,
                ),
            ],
            relation_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=compose(
                        torch.nn.init.normal_,
                        lambda x: 1.0e-03 * x,
                    ),
                ),
                EmbeddingSpecification(
                    shape=(embedding_dim, embedding_dim),
                    initializer=functools.partial(torch.nn.init.uniform_, a=-1, b=1),
                ),
            ],
            **kwargs,
        )
