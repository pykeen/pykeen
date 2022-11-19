# -*- coding: utf-8 -*-

"""A wrapper which combines an interaction function with NodePiece entity representations."""

import logging
from typing import Any, Callable, ClassVar, List, Mapping

import torch
from class_resolver import Hint, HintOrType, OptionalKwargs

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import NodePieceRepresentation, SubsetRepresentation, representation_resolver
from ...nn.modules import DistMultInteraction, Interaction
from ...nn.node_piece import RelationTokenizer, Tokenizer, tokenizer_resolver
from ...regularizers import Regularizer
from ...triples.triples_factory import CoreTriplesFactory
from ...typing import Constrainer, Initializer, Normalizer, OneOrSequence
from ...utils import upgrade_to_sequence

__all__ = [
    "NodePiece",
]

logger = logging.getLogger(__name__)


class NodePiece(ERModel):
    """A wrapper which combines an interaction function with NodePiece entity representations from [galkin2021]_.

    This model uses the :class:`pykeen.nn.NodePieceRepresentation` instead of a typical
    :class:`pykeen.nn.representation.Embedding` to more efficiently store representations.
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
        num_tokens: OneOrSequence[int] = 2,
        tokenizers: OneOrSequence[HintOrType[Tokenizer]] = None,
        tokenizers_kwargs: OneOrSequence[OptionalKwargs] = None,
        embedding_dim: int = 64,
        interaction: HintOrType[Interaction] = DistMultInteraction,
        aggregation: Hint[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        entity_initializer: Hint[Initializer] = None,
        entity_normalizer: Hint[Normalizer] = None,
        entity_constrainer: Hint[Constrainer] = None,
        entity_regularizer: Hint[Regularizer] = None,
        relation_initializer: Hint[Initializer] = None,
        relation_normalizer: Hint[Normalizer] = None,
        relation_constrainer: Hint[Constrainer] = None,
        relation_regularizer: Hint[Regularizer] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the triples factory. Must have create_inverse_triples set to True.
        :param num_tokens:
            the number of relations to use to represent each entity, cf.
            :class:`pykeen.nn.NodePieceRepresentation`.
        :param tokenizers:
            the tokenizer to use, cf. `pykeen.nn.node_piece.tokenizer_resolver`.
        :param tokenizers_kwargs:
            additional keyword-based parameters passed to the tokenizer upon construction.
        :param embedding_dim:
            the embedding dimension. Only used if embedding_specification is not given.
        :param interaction:
            the interaction module, or a hint for it.
        :param aggregation:
            aggregation of multiple token representations to a single entity representation. By default,
            this uses :func:`torch.mean`. If a string is provided, the module assumes that this refers to a top-level
            torch function, e.g. "mean" for :func:`torch.mean`, or "sum" for func:`torch.sum`. An aggregation can
            also have trainable parameters, .e.g., ``MLP(mean(MLP(tokens)))`` (cf. DeepSets from [zaheer2017]_). In
            this case, the module has to be created outside of this component.

            Moreover, we support providing "mlp" as a shortcut to use the MLP aggregation version from [galkin2021]_.

            We could also have aggregations which result in differently shapes output, e.g. a concatenation of all
            token embeddings resulting in shape ``(num_tokens * d,)``. In this case, `shape` must be provided.

            The aggregation takes two arguments: the (batched) tensor of token representations, in shape
            ``(*, num_tokens, *dt)``, and the index along which to aggregate.
        :param entity_initializer:
            a hint for initializing anchor embeddings
        :param entity_normalizer:
            a hint for normalizing anchor embeddings
        :param entity_constrainer:
            a hint for constraining anchor embeddings
        :param entity_regularizer:
            a hint for regularizing anchor embeddings
        :param relation_initializer:
            a hint for initializing relation embeddings
        :param relation_normalizer:
            a hint for normalizing relation embeddings
        :param relation_constrainer:
            a hint for constraining relation embeddings
        :param relation_regularizer:
            a hint for regularizing relation embeddings
        :param kwargs:
            additional keyword-based arguments passed to :meth:`ERModel.__init__`

        :raises ValueError:
            if the triples factory does not create inverse triples
        """
        if not triples_factory.create_inverse_triples:
            raise ValueError(
                "The provided triples factory does not create inverse triples. However, for the node piece "
                "representations inverse relation representations are required.",
            )

        # always create representations for normal and inverse relations and padding
        relation_representations = representation_resolver.make(
            query=None,
            max_id=2 * triples_factory.real_num_relations + 1,
            shape=embedding_dim,
            initializer=relation_initializer,
            normalizer=relation_normalizer,
            constrainer=relation_constrainer,
            regularizer=relation_regularizer,
        )

        # normalize embedding specification
        anchor_kwargs = dict(
            shape=embedding_dim,
            initializer=entity_initializer,
            normalizer=entity_normalizer,
            constrainer=entity_constrainer,
            regularizer=entity_regularizer,
        )

        # prepare token representations & kwargs
        token_representations = []
        token_representations_kwargs: List[OptionalKwargs] = []
        for tokenizer in upgrade_to_sequence(tokenizers):
            if tokenizer_resolver.lookup(tokenizer) is RelationTokenizer:
                token_representations.append(relation_representations)
                token_representations_kwargs.append(None)
            else:
                token_representations.append(None)  # Embedding
                token_representations_kwargs.append(anchor_kwargs)

        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=NodePieceRepresentation,
            entity_representations_kwargs=dict(
                triples_factory=triples_factory,
                token_representations=token_representations,
                token_representations_kwargs=token_representations_kwargs,
                tokenizers=tokenizers,
                tokenizers_kwargs=tokenizers_kwargs,
                aggregation=aggregation,
                num_tokens=num_tokens,
            ),
            relation_representations=SubsetRepresentation,
            relation_representations_kwargs=dict(  # hide padding relation
                # max_id=triples_factory.num_relations,  # will get added by ERModel
                base=relation_representations,
            ),
            **kwargs,
        )
