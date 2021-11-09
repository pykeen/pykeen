# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

import logging
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence

import torch
from class_resolver import Hint, HintOrType
from torch import nn

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import EmbeddingSpecification, NodePieceRepresentation
from ...nn.modules import DistMultInteraction, Interaction
from ...triples.triples_factory import CoreTriplesFactory

__all__ = [
    "NodePiece",
]

logger = logging.getLogger(__name__)


class _ConcatMLP(nn.Sequential):
    """A 2-layer MLP with ReLU activation and dropout applied to the concatenation of token representations.

    This is for conveniently choosing a configuration similar to the paper. For more complex aggregation mechanisms,
    pass an arbitrary callable instead.

    .. seealso:: https://github.com/migalkin/NodePiece/blob/d731c9990/lp_rp/pykeen105/nodepiece_rotate.py#L57-L65
    """

    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int,
        dropout: float = 0.1,
        ratio: int = 2,
    ):
        """
        Initialize the module.

        :param num_tokens:
            the number of tokens
        :param embedding_dim:
            the embedding dimension for a single token
        :param dropout:
            the dropout value on the hidden layer
        :param ratio:
            the ratio of the embedding dimension to the hidden layer size.
        """
        super().__init__(
            nn.Linear(num_tokens * embedding_dim, ratio * embedding_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(ratio * embedding_dim, embedding_dim),
        )

    def forward(self, xs: torch.FloatTensor, dim: int) -> torch.FloatTensor:  # noqa: D102
        # dim is only a parameter to match the signature of torch.mean / torch.sum
        # this class is not thought to be usable from outside
        assert dim == -2
        return super().forward(xs.view(*xs.shape[:-2], -1))


class NodePiece(ERModel):
    """A wrapper which combines an interaction function with NodePiece entity representations from [galkin2021]_.

    This model uses the :class:`pykeen.nn.emb.NodePieceRepresentation` instead of a typical
    :class:`pykeen.nn.emb.Embedding` to more efficiently store representations.
    ---
    citation:
        author: Galkin
        year: 2021
        link: https://arxiv.org/abs/2106.12144
        github: https://github.com/migalkin/NodePiece
    """

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        num_tokens: int = 2,
        embedding_dim: int = 64,
        embedding_specification: Optional[EmbeddingSpecification] = None,
        interaction: HintOrType[Interaction] = DistMultInteraction,
        aggregation: Hint[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        shape: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the triples factory used for tokenization
        :param embedding_dim:
            the embedding dimension. Only used if embedding_specification is not given.
        :param embedding_specification:
            the embedding specification.
        :param interaction:
            the interaction module, or a hint for it.
        :param aggregation:
            aggregation of multiple token representations to a single entity representation.
        :param shape:
            the shape of an individual representation. Only necessary, if aggregation results in a change of dimensions.
            this will only be necessary if the aggregation is an *ad hoc* function.
        :param kwargs:
            additional keyword-based arguments passed to :meth:`ERModel.__init__`
        """
        if not triples_factory.create_inverse_triples:
            logger.warning(
                "The provided triples factory does not create inverse triples. However, for the node piece"
                "representations inverse relation representations are required. Thus, the implicitly created inverse "
                "relations are only trained via the node piece mechanism, but not as part of the 'normal' training.",
            )
        embedding_specification = embedding_specification or EmbeddingSpecification(
            shape=(embedding_dim,),
        )

        # TODO put this inside :meth:`NodePieceRepresentation.__init__()`
        if aggregation == "mlp":
            aggregation = _ConcatMLP(
                num_tokens=num_tokens,
                embedding_dim=embedding_dim,
            )

        # always create representations for normal and inverse relations
        relation_representations = embedding_specification.make(
            num_embeddings=2 * triples_factory.real_num_relations,
        )
        entity_representations = NodePieceRepresentation(
            triples_factory=triples_factory,
            token_representation=relation_representations,
            aggregation=aggregation,
            shape=shape,
            k=num_tokens,
        )
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=entity_representations,
            relation_representations=relation_representations,
            **kwargs,
        )
