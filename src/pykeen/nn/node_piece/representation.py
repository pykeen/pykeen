# -*- coding: utf-8 -*-

"""Representation modules for NodePiece."""

import logging
import pathlib
from typing import Callable, Iterable, List, NamedTuple, Optional, Union

import torch
from class_resolver import HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs

from .tokenization import Tokenizer, tokenizer_resolver
from ..combination import ConcatAggregationCombination
from ..perceptron import ConcatMLP
from ..representation import CombinedRepresentation, Representation
from ..utils import ShapeError
from ...triples import CoreTriplesFactory
from ...typing import MappedTriples, OneOrSequence
from ...utils import broadcast_upgrade_to_sequences

__all__ = [
    "TokenizationRepresentation",
    "HashDiversityInfo",
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
        shape: Optional[OneOrSequence[int]] = None,
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
        :param shape:
            The shape of an individual representation. If provided, has to match.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`

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
    ) -> "TokenizationRepresentation":
        """
        Create a tokenization from applying a tokenizer.

        :param tokenizer:
            the tokenizer instance.
        :param num_tokens:
            the number of tokens to select for each entity.
        :param token_representation:
            the pre-instantiated token representations, class, or name of a class
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

    # docstr-coverage: inherited
    def iter_extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().iter_extra_repr()
        yield f"max_id={self.assignment.shape[0]}"
        yield f"num_tokens={self.num_tokens}"
        yield f"vocabulary_size={self.vocabulary_size}"

    # docstr-coverage: inherited
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

    @property
    def num_tokens(self) -> int:
        """Return the number of selected tokens for ID."""
        return self.assignment.shape[1]

    def save_assignment(self, output_path: pathlib.Path):
        """Save the assignment to a file.

        :param output_path:
            the output file path. Its parent directories will be created if necessary.
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
    uniques_per_representation: List[float]

    #: A scalar ratio of unique rows when combining all representations into one matrix, e.g. 0.95
    uniques_total: float


class NodePieceRepresentation(CombinedRepresentation):
    r"""
    Basic implementation of node piece decomposition [galkin2021]_.

    .. math ::
        x_e = agg(\{T[t] \mid t \in tokens(e) \})

    where $T$ are token representations, $tokens$ selects a fixed number of $k$ tokens for each entity, and $agg$ is
    an aggregation function, which aggregates the individual token representations to a single entity representation.
    """

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
        :param max_id:
            Only pass this to check if the number of entities in the triples factories is the same
        :param kwargs:
            additional keyword-based parameters passed to :meth:`CombinedRepresentation.__init__`
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
            combination_kwargs=dict(aggregation=aggregation, dim=-len(token_representations[0].shape)),
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

        .. code-block::

            from pykeen.model import NodePiece

            model = NodePiece(
                triples_factory=dataset.training,
                tokenizers=["AnchorTokenizer", "RelationTokenizer"],
                num_tokens=[20, 12],
                embedding_dim=64,
                interaction="rotate",
                relation_constrainer="complex_normalize",
                entity_initializer="xavier_uniform_",
            )
            print(model.entity_representations[0].estimate_diversity())

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
