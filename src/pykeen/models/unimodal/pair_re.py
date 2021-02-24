# -*- coding: utf-8 -*-

"""Implementation of PairRE."""

from typing import Any, ClassVar, Mapping, Optional

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss
from ...nn import EmbeddingSpecification
from ...nn.init import xavier_normal_
from ...nn.modules import PairREInteraction
from ...triples import TriplesFactory
from ...typing import DeviceHint, Hint, Initializer

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
        triples_factory: TriplesFactory,
        embedding_dim: int = 200,
        p: int = 2,
        power_norm: bool = True,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        entity_initializer: Hint[Initializer] = xavier_normal_,
        relation_initializer: Hint[Initializer] = xavier_normal_,
    ) -> None:
        r"""Initialize PairRE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param p: The $l_p$ norm
        :param power_norm: Should the power norm be used?
        
        .. warning ::
            Due to the lack of an official implementations, not all details are known.
        """
        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
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
            ]
        )
