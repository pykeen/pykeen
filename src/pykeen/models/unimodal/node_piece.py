# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

from typing import Any, Callable, ClassVar, Mapping, Optional, Union

import torch
from class_resolver.api import HintOrType
from torch import nn

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import EmbeddingSpecification, NodePieceRepresentation
from ...nn.modules import DistMultInteraction, Interaction
from ...triples.triples_factory import CoreTriplesFactory

__all__ = [
    "NodePiece",
]


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
        num_tokens: int,
        embedding_dim: int = 64,
        embedding_specification: Optional[EmbeddingSpecification] = None,
        relation_representations: Optional[EmbeddingSpecification] = None,
        interaction: HintOrType[Interaction] = DistMultInteraction,
        aggregation: Union[str, Callable[[torch.Tensor, int], torch.Tensor]] = None,
        node_piece_kwargs: Optional[Mapping[str, Any]] = None,
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
        :param relation_representations:
            the relation representations. Defaults to embedding_specification.
        :param interaction:
            the interaction module, or a hint for it.
        :param node_piece_kwargs:
            additional keyword-based arguments passed to the NodePieceRepresentation
        :param kwargs:
            additional keyword-based arguments passed to ERModel.__init__
        """
        embedding_specification = embedding_specification or EmbeddingSpecification(
            shape=(embedding_dim,),
        )
        node_piece_kwargs = node_piece_kwargs or {}
        # If it's already set, don't override
        node_piece_kwargs.setdefault("k", num_tokens)
        if aggregation == "mlp":
            # needs to be assigned to attribute to make sure that the trainable parameters are part of the model
            # parameters
            node_piece_kwargs["aggregation"] = _ConcatMLP(
                num_tokens=num_tokens,
                embedding_dim=embedding_dim,
            )
        entity_representations = NodePieceRepresentation(
            triples_factory=triples_factory,
            token_representation=embedding_specification,
            **node_piece_kwargs,
        )
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=entity_representations,
            relation_representations=relation_representations or embedding_specification,
            **kwargs,
        )
