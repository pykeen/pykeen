# -*- coding: utf-8 -*-

"""Implementation of MuRE."""

from typing import Any, ClassVar, Mapping, Optional

from torch.nn.init import normal_, uniform_, zeros_

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import EmbeddingSpecification
from ...nn.modules import MuREInteraction
from ...typing import Hint, Initializer

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
        entity_initializer: Hint[Initializer] = normal_,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_bias_initializer: Hint[Initializer] = zeros_,
        relation_initializer: Hint[Initializer] = normal_,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_matrix_initializer: Hint[Initializer] = uniform_,
        relation_matrix_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize MuRE via the :class:`pykeen.nn.modules.MuREInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$. Defaults to 200. Is usually $d \in [50, 300]$.
        :param p: The $l_p$ norm. Defaults to 2.
        :param power_norm: Should the power norm be used? Defaults to true.
        :param entity_initializer: Entity initializer function. Defaults to :func:`torch.nn.init.normal_`
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer
        :param entity_bias_initializer: Entity bias initializer function. Defaults to :func:`torch.nn.init.zeros_`
        :param relation_initializer: Relation initializer function. Defaults to :func:`torch.nn.init.normal_`
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer
        :param relation_matrix_initializer: Relation matrix initializer function.
            Defaults to :func:`torch.nn.init.uniform_`
        :param relation_matrix_initializer_kwargs: Keyword arguments to be used when calling the
            relation matrix initializer
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        # comment:
        # https://github.com/ibalazevic/multirelational-poincare/blob/34523a61ca7867591fd645bfb0c0807246c08660/model.py#L52
        # uses float64
        super().__init__(
            interaction=MuREInteraction,
            interaction_kwargs=dict(p=p, power_norm=power_norm),
            entity_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs or dict(
                        std=1.0e-03,
                    ),
                ),
                # entity bias for head
                EmbeddingSpecification(
                    shape=tuple(),  # scalar
                    initializer=entity_bias_initializer,
                ),
                # entity bias for tail
                EmbeddingSpecification(
                    shape=tuple(),  # scalar
                    initializer=entity_bias_initializer,
                ),
            ],
            relation_representations=[
                # relation offset
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs or dict(
                        std=1.0e-03,
                    ),
                ),
                # diagonal relation transformation matrix
                EmbeddingSpecification(
                    shape=(embedding_dim,),
                    initializer=relation_matrix_initializer,
                    initializer_kwargs=relation_matrix_initializer_kwargs or dict(
                        a=-1,
                        b=1,
                    ),
                ),
            ],
            **kwargs,
        )
