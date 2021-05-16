# -*- coding: utf-8 -*-

"""Implementation of PairRE."""

from typing import Any, ClassVar, Mapping, Optional

from torch.nn import functional
from torch.nn.init import uniform_

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import EmbeddingSpecification
from ...nn.modules import PairREInteraction
from ...typing import Hint, Initializer, Normalizer

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
        github: alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        p=dict(type=int, low=1, high=2),
    )

    #: The default entity normalizer parameters
    #: The entity representations are normalized to L2 unit length
    #: cf. https://github.com/alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE/blob/0a95bcd54759207984c670af92ceefa19dd248ad/biokg/model.py#L232-L240  # noqa: E501
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
        :param entity_initializer: Entity initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer
        :param entity_normalizer: Entity normalizer function. Defaults to :func:`torch.nn.functional.normalize`
        :param entity_normalizer_kwargs: Keyword arguments to be used when calling the entity normalizer
        :param relation_initializer: Relation initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        entity_normalizer_kwargs = _resolve_kwargs(
            kwargs=entity_normalizer_kwargs,
            default_kwargs=self.default_entity_normalizer_kwargs,
        )
        # update initializer settings, cf.
        # https://github.com/alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE/blob/0a95bcd54759207984c670af92ceefa19dd248ad/biokg/model.py#L45-L49
        # https://github.com/alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE/blob/0a95bcd54759207984c670af92ceefa19dd248ad/biokg/model.py#L29
        # https://github.com/alipay/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE/blob/0a95bcd54759207984c670af92ceefa19dd248ad/biokg/run.py#L50
        entity_initializer_kwargs = self._update_embedding_init_with_default(
            entity_initializer_kwargs,
            embedding_dim=embedding_dim,
        )
        relation_initializer_kwargs = self._update_embedding_init_with_default(
            relation_initializer_kwargs,
            # in the original implementation the embeddings are initialized in one parameter
            embedding_dim=2 * embedding_dim,
        )
        super().__init__(
            interaction=PairREInteraction,
            interaction_kwargs=dict(p=p, power_norm=power_norm),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
                initializer_kwargs=entity_initializer_kwargs,
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

    @staticmethod
    def _update_embedding_init_with_default(
        init_kwargs: Optional[Mapping[str, Any]],
        embedding_dim: int,
    ) -> Mapping[str, float]:
        """Update kwargs by dimension-based default init range."""
        init_kwargs = dict(init_kwargs or {})
        embedding_range = 14 / embedding_dim
        init_kwargs.setdefault("a", -embedding_range)
        init_kwargs.setdefault("b", embedding_range)
        return init_kwargs


def _resolve_kwargs(kwargs: Optional[Mapping[str, Any]], default_kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    kwargs = dict(kwargs or {})
    for k, v in default_kwargs.items():
        kwargs.setdefault(k, v)
    return kwargs
