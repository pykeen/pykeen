# -*- coding: utf-8 -*-

"""Implementation of PairRE."""

from typing import Any, ClassVar, Mapping, Optional

from torch.nn import functional
from torch.nn.init import uniform_

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import EmbeddingSpecification
from ...nn.modules import PairREInteraction
from ...typing import Hint, Initializer, Normalizer

__all__ = [
    'PairRE',
]


def _resolve_kwargs(kwargs: Optional[Mapping[str, Any]], default_kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    kwargs = kwargs or dict()
    for k, v in default_kwargs:
        kwargs.setdefault(k, v)
    return kwargs


class PairRE(ERModel):
    r"""An implementation of PairRE from [chao2020]_.

    ---
    citation:
        author: Chao
        year: 2020
        link: http://arxiv.org/abs/2011.03798
        github: alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        p=dict(type=int, low=1, high=2),
    )

    #: The default entity normalizer parameters
    # The entity representations are normalized to L2 unit length
    # cf. https://github.com/alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE/blob/0a95bcd54759207984c670af92ceefa19dd248ad/biokg/model.py#L232-L240
    default_entity_normalizer_kwargs: ClassVar[Mapping[str, Any]] = dict(
        p=2,
        dim=-1,
    )

    def __init__(
        self,
        embedding_dim: int = 200,
        p: int = 1,
        power_norm: bool = False,
        entity_initializer: Hint[Initializer] = uniform_,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_normalizer: Hint[Normalizer] = functional.normalize,
        entity_normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = uniform_,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize PairRE via the :class:`pykeen.nn.modules.PairREInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$.
        :param p: The $l_p$ norm.
        :param power_norm: Should the power norm be used?
        """
        entity_normalizer_kwargs = _resolve_kwargs(
            kwargs=entity_normalizer_kwargs,
            default_kwargs=self.default_entity_normalizer_kwargs,
        )
        # update initializer settings, cf.
        # https://github.com/alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE/blob/0a95bcd54759207984c670af92ceefa19dd248ad/biokg/model.py#L45-L49
        # https://github.com/alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE/blob/0a95bcd54759207984c670af92ceefa19dd248ad/biokg/model.py#L29
        # https://github.com/alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE/blob/0a95bcd54759207984c670af92ceefa19dd248ad/biokg/run.py#L50
        entity_initializer_kwargs = entity_initializer_kwargs or dict()
        embedding_range = 14 / embedding_dim
        entity_initializer_kwargs.setdefault("a", -embedding_range)
        entity_initializer_kwargs.setdefault("b", embedding_range)
        relation_initializer_kwargs = relation_initializer_kwargs or dict()
        relation_initializer_kwargs.setdefault("a", -embedding_range / 2)
        relation_initializer_kwargs.setdefault("b", embedding_range / 2)
        super().__init__(
            interaction=PairREInteraction(p=p, power_norm=power_norm),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
                normalizer=entity_normalizer,
                normalizer_kwargs=entity_normalizer_kwargs,
            ),
            relation_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
            ],
            **kwargs,
        )
