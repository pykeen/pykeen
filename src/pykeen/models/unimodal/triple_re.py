# -*- coding: utf-8 -*-

"""Implementation of TripleRE."""

from collections import ChainMap
from typing import Any, ClassVar, Mapping, Optional

from torch.nn import functional
from torch.nn.init import uniform_

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import EmbeddingSpecification
from ...nn.modules import TripleREInteraction
from ...typing import Hint, Initializer, Normalizer

__all__ = [
    "TripleRE",
]


class TripleRE(ERModel):
    r"""An implementation of TripleRE related to [yu2021]_.

    This implementation does not use NodePieceRepresentations so far.

    ---
    citation:
        author: Yu
        year: 2021
        link: https://vixra.org/abs/2112.0095
        github: https://github.com/LongYu-360/TripleRE-Add-NodePiece/
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        p=dict(type=int, low=1, high=2),
    )

    #: The default entity normalizer parameters
    #: The entity representations are normalized to L2 unit length
    #: cf. https://github.com/LongYu-360/TripleRE-Add-NodePiece/blob/994216dcb1d718318384368dd0135477f852c6a4/TripleRE%2BNodepiece/ogb_wikikg2/model.py#L323-L324
    default_entity_normalizer_kwargs: ClassVar[Mapping[str, Any]] = dict(
        p=2,
        dim=-1,
    )

    def __init__(
        self,
        embedding_dim: int = 128,
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
        r"""Initialize TripleRE via the :class:`pykeen.nn.modules.TripleREInteraction` interaction.

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
        # TODO: the published code uses NodePiece representations instead
        entity_normalizer_kwargs = ChainMap(entity_normalizer_kwargs, self.default_entity_normalizer_kwargs)
        super().__init__(
            interaction=TripleREInteraction,
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
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                ),
            ],
            **kwargs,
        )
