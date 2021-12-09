# -*- coding: utf-8 -*-

"""Implementation of CP model."""

from typing import Any, ClassVar, Mapping, Optional, Tuple, cast

import torch

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import EmbeddingSpecification
from ...nn.modules import CPInteraction
from ...typing import Hint, Initializer, Normalizer

__all__ = [
    "CP",
]


class CP(ERModel):
    r"""An implementation of CP as described in [lacroix2018]_ based on [hitchcock1927]_.

    ---
    name: Canonical Tensor Decomposition
    citation:
        author: Lacroix
        year: 2018
        arxiv: 1806.07297
        link: https://arxiv.org/abs/1806.07297
        github: facebookresearch/kbc
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE, rank=dict(type=int, low=2, high=2048, log=True)
    )

    def __init__(
        self,
        embedding_dim: int = 64,
        rank: int = 64,
        entity_initializer: Hint[Initializer] = None,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_normalizer: Hint[Normalizer] = None,
        entity_normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = None,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize CP via the :class:`pykeen.nn.modules.CPInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$.
        :param rank: The tensor decomposition rank $k$.
        :param entity_initializer: Entity initializer function. Defaults to None
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer
        :param entity_normalizer: Entity normalizer function. Defaults to None
        :param entity_normalizer_kwargs: Keyword arguments to be used when calling the entity normalizer
        :param relation_initializer: Relation initializer function. Defaults to None
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=CPInteraction,
            entity_representations=[
                # head representation
                EmbeddingSpecification(
                    shape=(rank, embedding_dim),
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs,
                    normalizer=entity_normalizer,
                    normalizer_kwargs=entity_normalizer_kwargs,
                ),
                # tail representation
                EmbeddingSpecification(
                    shape=(rank, embedding_dim),
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs,
                    normalizer=entity_normalizer,
                    normalizer_kwargs=entity_normalizer_kwargs,
                ),
            ],
            relation_representations=EmbeddingSpecification(
                shape=(rank, embedding_dim),
                initializer=relation_initializer,
                initializer_kwargs=relation_initializer_kwargs,
            ),
            # Since CP uses different representations for entities in head / tail role,
            # the current solution is a bit hacky, and may be improved. See discussion
            # on https://github.com/pykeen/pykeen/pull/663.
            skip_checks=True,
            **kwargs,
        )

    def _get_representations(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:  # noqa: D102
        # Override to allow different head and tail entity representations
        h, r, t = [
            [representation.get_in_more_canonical_shape(dim=dim, indices=indices) for representation in representations]
            for dim, indices, representations in (
                ("h", h_indices, self.entity_representations[0:1]),  # <== this is different
                ("r", r_indices, self.relation_representations),
                ("t", t_indices, self.entity_representations[1:2]),  # <== this is different
            )
        ]
        # normalization
        return cast(
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
            tuple(x[0] if len(x) == 1 else x for x in (h, r, t)),
        )
