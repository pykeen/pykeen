# -*- coding: utf-8 -*-

"""Implementation of the BoxE model."""

from typing import Any, ClassVar, Mapping, Optional

from torch.nn.init import uniform_

from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import NSSALoss
from ...models import ERModel
from ...nn.init import uniform_norm_
from ...nn.modules import BoxEInteraction
from ...typing import Hint, Initializer

__all__ = [
    "BoxE",
]


class BoxE(ERModel):
    r"""An implementation of BoxE from [abboud2020]_.

    .. note::

        This implementation only currently supports unimodal knowledge graphs consisting only of binary facts,
        whereas the original BoxE applies to arbitrary facts of any arity, i.e., unary facts, binary facts,
        ternary facts, etc. For use on higher-arity knowledge bases, please refer to the original implementation at
        https://www.github.com/ralphabb/BoxE.

    ---
    citation:
        author: Abboud
        year: 2020
        link: https://arxiv.org/abs/2007.06267
        github: ralphabb/BoxE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        p=dict(type=int, low=1, high=2),
    )

    loss_default = NSSALoss
    loss_default_kwargs = dict(margin=3, adversarial_temperature=2.0, reduction="sum")

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        tanh_map: bool = True,
        p: int = 2,
        power_norm: bool = False,
        entity_initializer: Hint[Initializer] = uniform_norm_,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = uniform_norm_,  # Has to be scaled as well
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_size_initializer: Hint[Initializer] = uniform_,  # Has to be scaled as well
        relation_size_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize BoxE.

        :param embedding_dim:
            The entity embedding dimension $d$. Defaults to 200. Is usually $d \in [50, 300]$.
        :param tanh_map:
            Whether to use tanh mapping after BoxE computation (defaults to true). The hyperbolic tangent mapping
            restricts the embedding space to the range [-1, 1], and thus this map implicitly
            regularizes the space to prevent loss reduction by growing boxes arbitrarily large.
        :param p:
            order of norm in score computation
        :param power_norm:
            whether to use the p-th power of the norm instead
        :param entity_initializer:
            Entity initializer function. Defaults to :func:`pykeen.nn.init.uniform_norm_`
        :param entity_initializer_kwargs:
            Keyword arguments to be used when calling the entity initializer
        :param relation_initializer:
            Relation initializer function. Defaults to :func:`pykeen.nn.init.uniform_norm_`
        :param relation_initializer_kwargs:
            Keyword arguments to be used when calling the relation initializer
        :param relation_size_initializer:
            Relation initializer function. Defaults to :func:`torch.nn.init.uniform_`
            Defaults to :func:`torch.nn.init.uniform_`
        :param relation_size_initializer_kwargs: Keyword arguments to be used when calling the
            relation matrix initializer
        :param kwargs:
            Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.

        This interaction relies on Abboud's point-to-box distance
        :func:`pykeen.utils.point_to_box_distance`.
        """
        super().__init__(
            interaction=BoxEInteraction,
            interaction_kwargs=dict(
                p=p,
                power_norm=power_norm,
                tanh_map=tanh_map,
            ),
            entity_representations_kwargs=[  # Base position
                dict(
                    shape=embedding_dim,
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs,
                ),  # Bump
                # entity bias for head
                dict(
                    shape=embedding_dim,
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs,
                ),
            ],
            relation_representations_kwargs=[
                # relation position head
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
                # relation shape head
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
                # relation size head
                dict(
                    shape=(1,),
                    initializer=relation_size_initializer,
                    initializer_kwargs=relation_size_initializer_kwargs,
                ),
                # relation position tail
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
                # relation shape tail
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
                # relation size tail
                dict(
                    shape=(1,),
                    initializer=relation_size_initializer,
                    initializer_kwargs=relation_size_initializer_kwargs,
                ),
            ],
            **kwargs,
        )
