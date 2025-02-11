"""Representation modules for NodePiece."""

import logging
import pathlib
from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch
from class_resolver import (
    HintOrType,
    OneOrManyHintOrType,
    OneOrManyOptionalKwargs,
    OptionalKwargs,
    ResolverKey,
    update_docstring_with_resolver_keys,
)
from docdata import parse_docdata
from typing_extensions import Self

from .tokenization import Tokenizer, tokenizer_resolver
from ..combination import ConcatAggregationCombination
from ..perceptron import ConcatMLP
from ..representation import CombinedRepresentation, Representation
from ..utils import ShapeError
from ...triples import CoreTriplesFactory
from ...typing import FloatTensor, LongTensor, MappedTriples, OneOrSequence
from ...utils import broadcast_upgrade_to_sequences

__all__ = [
    "TokenizationRepresentation",
    "HashDiversityInfo",
    "NodePieceRepresentation",
]

logger = logging.getLogger(__name__)


@parse_docdata
class TokenizationRepresentation(Representation):
    r"""A module holding the result of tokenization.

    It represents each index by the concatenation of representations of the corresponding tokens.

    .. math ::
        [T[t] | t \in \textit{tok}(i)]

    where $tok(i)$ denotes the sequence of token indices for the given index $i$,
    and $T$ stores the representations for each token.

    ---
    name: Tokenization
    citation:
        author: Galkin
        year: 2021
        arxiv: 2106.12144
        link: https://arxiv.org/abs/2106.12144
        github: migalkin/NodePiece
    """

    #: the token ID of the padding token
    vocabulary_size: int

    #: the token representations
    vocabulary: Representation

    #: the assigned tokens for each entity
    assignment: LongTensor

    @update_docstring_with_resolver_keys(
        ResolverKey(name="token_representation", resolver="pykeen.nn.representation_resolver")
    )
    def __init__(
        self,
        assignment: LongTensor,
        token_representation: HintOrType[Representation] = None,
        token_representation_kwargs: OptionalKwargs = None,
        shape: OneOrSequence[int] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the tokenization.

        :param assignment: shape: ``(n, num_chosen_tokens)``
            The token assignment.
        :param token_representation: shape: ``(num_total_tokens, *shape)``
            The token representations.
        :param token_representation_kwargs:
            Additional keyword-based parameters.
        :param shape:
            The shape of an individual representation. If provided, has to match
            ``(assignment.shape[1], *token_representation.shape)``.
        :param kwargs:
            Additional keyword-based parameters passed to :class:`~pykeen.nn.representation.Representation`.

        :raises ValueError:
            If there's a mismatch between the representation size and the vocabulary size.
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
        shape = ShapeError.verify(shape=(num_chosen_tokens,) + token_representation.shape, reference=shape)
        super().__init__(max_id=max_id, shape=shape, **kwargs)

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
    ) -> Self:
        """
        Create a tokenization from applying a tokenizer.

        :param tokenizer:
            The tokenizer instance.
        :param num_tokens:
            The number of tokens to select for each entity.
        :param token_representation: shape: ``(num_total_tokens, *shape)``
            The token representations.
        :param token_representation_kwargs:
            Additional keyword-based parameters.
        :param mapped_triples:
            The ID-based triples.
        :param num_entities:
            The number of entities.
        :param num_relations:
            The number of relations.
        :param kwargs:
            Additional keyword-based parameters passed to :class:`~pykeen.nn.node_piece.TokenizationRepresentation`.

        :return:
            A :class:`~pykeen.nn.node_piece.TokenizationRepresentation` by applying the tokenizer.
        """
        # apply tokenizer
        assignment = tokenizer(
            mapped_triples=mapped_triples,
            num_tokens=num_tokens,
            num_entities=num_entities,
            num_relations=num_relations,
        )[1]
        return cls(
            assignment=assignment,
            token_representation=token_representation,
            token_representation_kwargs=token_representation_kwargs,
            **kwargs,
        )

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"max_id={self.assignment.shape[0]}"
        yield f"num_tokens={self.num_tokens}"
        yield f"vocabulary_size={self.vocabulary_size}"

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: LongTensor | None = None,
    ) -> FloatTensor:  # noqa: D102
        # get token IDs, shape: (*, num_chosen_tokens)
        token_ids = self.assignment
        if indices is not None:
            token_ids = token_ids[indices]

        # lookup token representations, shape: (*, num_chosen_tokens, *shape)
        return self.vocabulary(token_ids)

    @property
    def num_tokens(self) -> int:
        """Return the number of selected tokens for each index."""
        return self.assignment.shape[1]

    def save_assignment(self, output_path: pathlib.Path):
        """Save the assignment to a file.

        :param output_path:
            The output file path. Its parent directories will be created if necessary.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.assignment, output_path)
        logger.info(f"Saved assignment of shape {self.assignment.shape} to {output_path}")


class HashDiversityInfo(NamedTuple):
    """A ratio information object.

    A pair `unique_per_repr, unique_total`, where `unique_per_repr` is a list with
    the percentage of unique hashes for each token representation, and `unique_total`
    the frequency of unique hashes when we concatenate all token representations.
    """

    #: A list with ratios per representation in their creation order,
    #: e.g., ``[0.58, 0.82]`` for :class:`AnchorTokenization` and :class:`RelationTokenization`
    uniques_per_representation: list[float]

    #: A scalar ratio of unique rows when combining all representations into one matrix, e.g. 0.95
    uniques_total: float


@parse_docdata
class NodePieceRepresentation(CombinedRepresentation):
    r"""
    Basic implementation of NodePiece decomposition [galkin2021]_.

    .. math ::
        x_e = \textit{agg}(\{T[t] \mid t \in \textit{tok}(e) \})

    where $T$ are token representations, *tok* selects a fixed number of $k$ tokens for each index, and *agg* is
    an aggregation function, which aggregates the individual token representations to a single representation.

    ---
    name: NodePiece
    citation:
        author: Galkin
        year: 2021
        link: https://arxiv.org/abs/2106.12144
        github: https://github.com/migalkin/NodePiece
    """

    @update_docstring_with_resolver_keys(
        ResolverKey("token_representations", resolver="pykeen.nn.representation_resolver"),
        ResolverKey("tokenizers", resolver="pykeen.nn.node_piece.tokenizer_resolver"),
        ResolverKey("aggregation", resolver="class_resolver.contrib.torch.aggregation_resolver"),
    )
    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        token_representations: OneOrManyHintOrType[Representation] = None,
        token_representations_kwargs: OneOrManyOptionalKwargs = None,
        tokenizers: OneOrManyHintOrType[Tokenizer] = None,
        tokenizers_kwargs: OneOrManyOptionalKwargs = None,
        num_tokens: OneOrSequence[int] = 2,
        aggregation: None | str | Callable[[FloatTensor, int], FloatTensor] = None,
        aggregation_kwargs: OptionalKwargs = None,
        max_id: int | None = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        :param triples_factory:
            The triples factory, required for tokenization.

        :param token_representations:
            The token representation specification, or pre-instantiated representation module.
        :param token_representations_kwargs:
            Additional keyword-based parameters.

        :param tokenizers:
            The tokenizer to use.
        :param tokenizers_kwargs:
            Additional keyword-based parameters passed to the tokenizer upon construction.

        :param num_tokens:
            The number of tokens for each entity.
        :param aggregation:
            Aggregation of multiple token representations to a single entity representation. By default,
            this uses :func:`torch.mean`. If a string is provided, the module assumes that this refers to a top-level
            torch function, e.g. "mean" for :func:`torch.mean`, or "sum" for func:`torch.sum`. An aggregation can
            also have trainable parameters, .e.g., ``MLP(mean(MLP(tokens)))`` (cf. DeepSets from [zaheer2017]_). In
            this case, the module has to be created outside of this component.

            We could also have aggregations which result in differently shapes output, e.g. a concatenation of all
            token embeddings resulting in shape ``(num_tokens * d,)``. In this case, `shape` must be provided.

            The aggregation takes two arguments: the (batched) tensor of token representations, in shape
            ``(*, num_tokens, *dt)``, and the index along which to aggregate.
        :param aggregation_kwargs:
            Additional keyword-based parameters.
        :param max_id:
            Only pass this to check if the number of entities in the triples factories is the same.
        :param kwargs:
            Additional keyword-based parameters passed to :class:`~pykeen.nn.representation.CombinedRepresentation`.
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
                strict=False,
            )
        ]

        # Create an MLP for string aggregation
        if aggregation == "mlp":
            # note: the token representations' shape includes the number of tokens as leading dim
            embedding_dim = token_representations[0].shape[1]
            aggregation = ConcatMLP(
                input_dim=embedding_dim * sum(num_tokens),
                output_dim=embedding_dim,
            )

        super().__init__(
            max_id=triples_factory.num_entities,
            base=token_representations,
            combination=ConcatAggregationCombination,
            combination_kwargs=dict(
                aggregation=aggregation, aggregation_kwargs=aggregation_kwargs, dim=-len(token_representations[0].shape)
            ),
            **kwargs,
        )

    def estimate_diversity(self) -> HashDiversityInfo:
        """
        Estimate the diversity of the tokens via their hashes.

        :return:
            A ratio information tuple

        Tokenization strategies might produce exactly the same hashes for
        several nodes depending on the graph structure and tokenization
        parameters. Same hashes will result in same node representations
        and, hence, might inhibit the downstream performance.
        This function comes handy when you need to estimate the diversity
        of built node hashes under a certain tokenization strategy - ideally,
        you'd want every node to have a unique hash.
        The function computes how many node hashes are unique in each
        representation and overall (if we concat all of them in a single row).
        1.0 means that all nodes have unique hashes.

        Example usage:

        .. literalinclude:: ../examples/nn/representation/node_piece_diversity.py

        .. seealso:: https://github.com/pykeen/pykeen/pull/896
        """
        # unique hashes per representation
        uniques_per_representation = [tokens.assignment.unique(dim=0).shape[0] / self.max_id for tokens in self.base]

        # unique hashes if we concatenate all representations together
        unnormalized_uniques_total = torch.unique(
            torch.cat([tokens.assignment for tokens in self.base], dim=-1), dim=0
        ).shape[0]
        return HashDiversityInfo(
            uniques_per_representation=uniques_per_representation,
            uniques_total=unnormalized_uniques_total / self.max_id,
        )
