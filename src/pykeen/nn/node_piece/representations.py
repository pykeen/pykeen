# -*- coding: utf-8 -*-

"""Representation modules for NodePiece."""

import logging
from typing import Callable, Optional, Sequence, Union

import torch
from class_resolver import HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs
from class_resolver.contrib.torch import aggregation_resolver

from .tokenization import Tokenizer, tokenizer_resolver
from ..representation import Representation
from ...triples import CoreTriplesFactory
from ...typing import MappedTriples, OneOrSequence
from ...utils import broadcast_upgrade_to_sequences

__all__ = [
    "TokenizationRepresentation",
    "NodePieceRepresentation",
]

logger = logging.getLogger(__name__)


class TokenizationRepresentation(Representation):
    """A module holding the result of tokenization."""

    #: the token ID of the padding token
    vocabulary_size: int

    #: the token representations
    vocabulary: Representation

    #: the assigned tokens for each entity
    assignment: torch.LongTensor

    def __init__(
        self,
        assignment: torch.LongTensor,
        token_representation: HintOrType[Representation] = None,
        token_representation_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """
        Initialize the tokenization.

        :param assignment: shape: `(n, num_chosen_tokens)`
            the token assignment.
        :param token_representation: shape: `(num_total_tokens, *shape)`
            the token representations
        :param token_representation_kwargs:
            additional keyword-based parameters
        :param kwargs:
            additional keyword-based parameters passed to super.__init__
        :raises ValueError: if there's a mismatch between the representation size
            and the vocabulary size
        """
        # needs to be lazily imported to avoid cyclic imports
        from .. import representation_resolver

        # fill padding (nn.Embedding cannot deal with negative indices)
        padding = assignment < 0
        # sometimes, assignment.max() does not cover all relations (eg, inductive inference graphs
        # contain a subset of training relations) - for that, the padding index is the last index of the Representation
        self.vocabulary_size = (
            token_representation.max_id
            if isinstance(token_representation, Representation)
            else assignment.max().item() + 2  # exclusive (+1) and including padding (+1)
        )

        assignment[padding] = self.vocabulary_size - 1  # = assignment.max().item() + 1
        max_id, num_chosen_tokens = assignment.shape

        # resolve token representation
        token_representation = representation_resolver.make(
            token_representation,
            token_representation_kwargs,
            max_id=self.vocabulary_size,
        )
        super().__init__(max_id=max_id, shape=(num_chosen_tokens,) + token_representation.shape, **kwargs)

        # input validation
        if token_representation.max_id < self.vocabulary_size:
            raise ValueError(
                f"The token representations only contain {token_representation.max_id} representations,"
                f"but there are {self.vocabulary_size} tokens in use.",
            )
        elif token_representation.max_id > self.vocabulary_size:
            logger.warning(
                f"Token representations do contain more representations ({token_representation.max_id}) "
                f"than tokens are used ({self.vocabulary_size}).",
            )
        # register as buffer
        self.register_buffer(name="assignment", tensor=assignment)
        # assign sub-module
        self.vocabulary = token_representation

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Tokenizer,
        num_tokens: int,
        mapped_triples: MappedTriples,
        num_entities: int,
        num_relations: int,
        token_representation: HintOrType[Representation] = None,
        token_representation_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> "TokenizationRepresentation":
        """
        Create a tokenization from applying a tokenizer.

        :param tokenizer:
            the tokenizer instance.
        :param num_tokens:
            the number of tokens to select for each entity.
        :param token_representation:
            the pre-instantiated token representations, or an EmbeddingSpecification to create them
        :param token_representation_kwargs:
            additional keyword-based parameters
        :param mapped_triples:
            the ID-based triples
        :param num_entities:
            the number of entities
        :param num_relations:
            the number of relations
        :param kwargs:
            additional keyword-based parameters passed to TokenizationRepresentation.__init__
        :return:
            A tokenization representation by applying the tokenizer
        """
        # apply tokenizer
        vocabulary_size, assignment = tokenizer(
            mapped_triples=mapped_triples,
            num_tokens=num_tokens,
            num_entities=num_entities,
            num_relations=num_relations,
        )
        return TokenizationRepresentation(
            assignment=assignment,
            token_representation=token_representation,
            token_representation_kwargs=token_representation_kwargs,
            **kwargs,
        )

    def extra_repr(self) -> str:  # noqa: D102
        return "\n".join(
            (
                f"max_id={self.assignment.shape[0]},",
                f"num_tokens={self.assignment.shape[1]},",
                f"vocabulary_size={self.vocabulary_size},",
            )
        )

    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # get token IDs, shape: (*, num_chosen_tokens)
        token_ids = self.assignment
        if indices is not None:
            token_ids = token_ids[indices]

        # lookup token representations, shape: (*, num_chosen_tokens, *shape)
        return self.vocabulary(token_ids)


class NodePieceRepresentation(Representation):
    r"""
    Basic implementation of node piece decomposition [galkin2021]_.

    .. math ::
        x_e = agg(\{T[t] \mid t \in tokens(e) \})

    where $T$ are token representations, $tokens$ selects a fixed number of $k$ tokens for each entity, and $agg$ is
    an aggregation function, which aggregates the individual token representations to a single entity representation.

    .. note ::
        This implementation currently only supports representation of entities by bag-of-relations.
    """

    #: the token representations
    token_representations: Sequence[TokenizationRepresentation]

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        token_representations: OneOrManyHintOrType[Representation] = None,
        token_representations_kwargs: OneOrManyOptionalKwargs = None,
        tokenizers: OneOrManyHintOrType[Tokenizer] = None,
        tokenizers_kwargs: OneOrManyOptionalKwargs = None,
        num_tokens: OneOrSequence[int] = 2,
        aggregation: Union[None, str, Callable[[torch.FloatTensor, int], torch.FloatTensor]] = None,
        max_id: Optional[int] = None,
        shape: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        :param triples_factory:
            the triples factory
        :param token_representations:
            the token representation specification, or pre-instantiated representation module.
        :param token_representations_kwargs:
            additional keyword-based parameters
        :param tokenizers:
            the tokenizer to use, cf. `pykeen.nn.node_piece.tokenizer_resolver`.
        :param tokenizers_kwargs:
            additional keyword-based parameters passed to the tokenizer upon construction.
        :param num_tokens:
            the number of tokens for each entity.
        :param aggregation:
            aggregation of multiple token representations to a single entity representation. By default,
            this uses :func:`torch.mean`. If a string is provided, the module assumes that this refers to a top-level
            torch function, e.g. "mean" for :func:`torch.mean`, or "sum" for func:`torch.sum`. An aggregation can
            also have trainable parameters, .e.g., ``MLP(mean(MLP(tokens)))`` (cf. DeepSets from [zaheer2017]_). In
            this case, the module has to be created outside of this component.

            We could also have aggregations which result in differently shapes output, e.g. a concatenation of all
            token embeddings resulting in shape ``(num_tokens * d,)``. In this case, `shape` must be provided.

            The aggregation takes two arguments: the (batched) tensor of token representations, in shape
            ``(*, num_tokens, *dt)``, and the index along which to aggregate.
        :param shape:
            the shape of an individual representation. Only necessary, if aggregation results in a change of dimensions.
            this will only be necessary if the aggregation is an *ad hoc* function.
        :param max_id:
            Only pass this to check if the number of entities in the triples factories is the same
        :param kwargs:
            additional keyword-based parameters passed to super.__init__
        :raises ValueError: if the shapes for any vocabulary entry
            in all token representations are inconsistent
        """
        if max_id:
            assert max_id == triples_factory.num_entities

        # normalize triples
        mapped_triples = triples_factory.mapped_triples
        if triples_factory.create_inverse_triples:
            # inverse triples are created afterwards implicitly
            mapped_triples = mapped_triples[mapped_triples[:, 1] < triples_factory.real_num_relations]

        token_representations, token_representations_kwargs, num_tokens = broadcast_upgrade_to_sequences(
            token_representations, token_representations_kwargs, num_tokens
        )

        # tokenize
        token_representations = [
            TokenizationRepresentation.from_tokenizer(
                tokenizer=tokenizer_inst,
                num_tokens=num_tokens_,
                token_representation=token_representation,
                token_representation_kwargs=token_representation_kwargs,
                mapped_triples=mapped_triples,
                num_entities=triples_factory.num_entities,
                num_relations=triples_factory.real_num_relations,
            )
            for tokenizer_inst, token_representation, token_representation_kwargs, num_tokens_ in zip(
                tokenizer_resolver.make_many(queries=tokenizers, kwargs=tokenizers_kwargs),
                token_representations,
                token_representations_kwargs,
                num_tokens,
            )
        ]

        # determine shape
        if shape is None:
            shapes = {t.vocabulary.shape for t in token_representations}
            if len(shapes) != 1:
                raise ValueError(f"Inconsistent token shapes: {shapes}")
            shape = list(shapes)[0]

        # super init; has to happen *before* any parameter or buffer is assigned
        super().__init__(max_id=triples_factory.num_entities, shape=shape, **kwargs)

        # assign module
        self.token_representations = torch.nn.ModuleList(token_representations)

        # Assign default aggregation
        self.aggregation = aggregation_resolver.lookup(aggregation)
        self.aggregation_index = -(1 + len(shape))

    def extra_repr(self) -> str:  # noqa: D102
        aggregation_str = self.aggregation.__name__ if hasattr(self.aggregation, "__name__") else str(self.aggregation)
        return f"aggregation={aggregation_str}, "

    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        return self.aggregation(
            torch.cat(
                [tokenization(indices=indices) for tokenization in self.token_representations],
                dim=self.aggregation_index,
            ),
            self.aggregation_index,
        )
