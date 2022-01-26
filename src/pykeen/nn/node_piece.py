"""Node Piece representations."""
import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Optional, Sequence, Union

import numpy
import scipy.sparse
import scipy.sparse.csgraph
import torch
from class_resolver import HintOrType, OptionalKwargs, Resolver

from .emb import EmbeddingSpecification, RepresentationModule
from ..constants import AGGREGATIONS
from ..triples import CoreTriplesFactory
from ..typing import MappedTriples

__all__ = [
    "NodePieceRepresentation",
]

logger = logging.getLogger(__name__)


def _sample(rs: torch.LongTensor, k: int) -> torch.LongTensor:
    """Sample without replacement."""
    return rs[torch.randperm(rs.shape[0])[:k]]


class Tokenizer:
    """A base class for tokenizers for NodePiece representations."""

    @abstractmethod
    def __call__(
        self,
        mapped_triples: MappedTriples,
        num_tokens: int,
        num_entities: int,
        num_relations: int,
    ) -> torch.LongTensor:
        """
        Tokenize the entities contained given the triples.

        :param mapped_triples: shape: (n, 3)
            the ID-based triples
        :param num_tokens:
            the number of tokens to select for each entity
        :param num_entities:
            the number of entities
        :param num_relations:
            the number of relatiosn

        :return: shape: (num_entities, num_tokens), -1 <= res < max_token_id
            the selected relation IDs for each entity. -1 is used as a padding token.
        """
        raise NotImplementedError


class RelationTokenizer(Tokenizer):
    """Tokenize entities by representing them as a bag of relations."""

    def __call__(
        self,
        mapped_triples: MappedTriples,
        num_tokens: int,
        num_entities: int,
        num_relations: int,
    ) -> torch.LongTensor:  # noqa: D102
        # tokenize: represent entities by bag of relations
        h, r, t = mapped_triples.t()

        # collect candidates
        e2r = defaultdict(set)
        for e, r in (
            torch.cat(
                [
                    torch.stack([h, r], dim=1),
                    torch.stack([t, r + num_relations], dim=1),
                ],
                dim=0,
            )
            .unique(dim=0)
            .tolist()
        ):
            e2r[e].add(r)

        # randomly sample without replacement num_tokens relations for each entity
        assignment = torch.full(
            size=(num_entities, num_tokens),
            dtype=torch.long,
            fill_value=-1,
        )
        for e, rs in e2r.items():
            rs = torch.as_tensor(data=list(rs), dtype=torch.long)
            rs = _sample(rs=rs, k=num_tokens)
            assignment[e, : len(rs)] = rs

        return assignment


class AnchorSelection:
    """Anchor entity selection strategy."""

    def __init__(self, num_anchors: int = 32) -> None:
        """
        Initialize the strategy.

        :param num_anchors:
            the number of anchor nodes to select.
            # TODO: allow relative
        """
        self.num_anchors = num_anchors

    @abstractmethod
    def __call__(self, edge_index: numpy.ndarray) -> numpy.ndarray:
        """
        Select anchor nodes.

        .. note ::
            the number of selected anchors may be smaller than $k$, if there
            are less entities present in the edge index.

        :param edge_index: shape: (m, 2)
            the edge_index, i.e., adjacency list.

        :return: (k,)
            the selected entity ids
        """
        raise NotImplementedError


class DegreeAnchorSelection(AnchorSelection):
    """Select entities according to their (undirected) degree."""

    def __call__(self, edge_index: numpy.ndarray) -> numpy.ndarray:  # noqa: D102
        unique, counts = numpy.unique(edge_index, return_counts=True)
        top_ids = numpy.argpartition(counts, max(counts.size - self.num_anchors, 0))[-self.num_anchors :]
        return unique[top_ids]


anchor_selection_resolver: Resolver[AnchorSelection] = Resolver.from_subclasses(
    base=AnchorSelection,
    default=DegreeAnchorSelection,
)


class AnchorSearcher:
    """A method for finding the closest anchors."""

    @abstractmethod
    def __call__(self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int) -> numpy.ndarray:
        """
        Find the $k$ closest anchor nodes for each entity.

        :param edge_index: shape: (2, m)
            the edge index
        :param anchors: shape: (a,)
            the selected anchor entity Ids
        :param k:
            the number of closest anchors to return

        :return: shape: (n, k), -1 <= res < a
            the Ids of the closest anchors
        """
        raise NotImplementedError


class CSGraphAnchorSearcher(AnchorSearcher):
    """Find closest anchors using scipy.sparse.csgraph."""

    def __call__(self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int) -> numpy.ndarray:  # noqa: D102
        num_entities = edge_index.max().item() + 1
        adjacency = scipy.sparse.coo_matrix(
            (
                numpy.ones_like(edge_index[0], dtype=bool),
                tuple(edge_index),
            ),
            shape=(num_entities, num_entities),
        )
        # compute distances between anchors and all nodes, shape: (num_anchors, num_entities)
        distances = scipy.sparse.csgraph.shortest_path(
            csgraph=adjacency,
            directed=False,
            return_predecessors=False,
            unweighted=True,
            indices=anchors,
        )
        # select anchor IDs with smallest distance
        return torch.as_tensor(
            numpy.argpartition(distances, kth=min(k, num_entities), axis=0)[:k, :].T,
            dtype=torch.long,
        )


class ScipySparseAnchorSearcher(AnchorSearcher):
    """Find closest anchors using scipy.sparse."""

    def __init__(self, max_iter: int = 5) -> None:
        self.max_iter = max_iter

    def __call__(self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int) -> numpy.ndarray:  # noqa: D102
        # infer shape
        num_entities = edge_index.max().item() + 1
        # create adjacency matrix
        adjacency = scipy.sparse.coo_matrix(
            (
                numpy.ones_like(edge_index[0], dtype=bool),
                tuple(edge_index),
            ),
            shape=(num_entities, num_entities),
        )
        # symmetric + self-loops
        adjacency = adjacency + adjacency.transpose() + scipy.sparse.eye(num_entities, dtype=bool, format="coo")
        adjacency = adjacency.tocsr()
        logger.info(f"Created adjacency matrix: {adjacency}")

        # for each entity, determine anchor pool by BFS
        num_anchors = len(anchors)
        pool = numpy.zeros(shape=(num_entities, num_anchors), dtype=bool)
        reachable = numpy.zeros(shape=(num_entities, num_anchors), dtype=bool)
        reachable[anchors] = numpy.eye(num_anchors, dtype=bool)
        final = numpy.zeros(shape=(num_entities,), dtype=bool)

        for _i in range(self.max_iter):
            # propagate one hop
            reachable = adjacency.dot(reachable)
            # copy pool if we have seen enough anchors and have not yet stopped
            num_reachable = reachable.sum(axis=1)
            enough = num_reachable >= k
            mask = enough & ~final
            pool[mask] = reachable[mask]
            # stop once we have enough
            final |= enough
        del reachable, final

        tokens = numpy.full(shape=(num_entities, k), fill_value=-1)
        # select from pool
        # use sparse random sampling due to memory footprint
        entity_ids, anchor_ids = pool.nonzero()
        unique_entity_ids, counts = numpy.unique(entity_ids, return_counts=True)
        assert (counts >= k).all()
        generator = numpy.random.default_rng()
        intra_offset = numpy.floor(
            generator.random(size=(counts.size, k), dtype=numpy.float32) * counts[:, None]
        ).astype(int)
        offset = numpy.cumsum(numpy.r_[0, counts])[:-1, None] + intra_offset
        tokens[unique_entity_ids] = anchor_ids[offset]

        return tokens


# TODO: use graph library, such as igraph, graph-tool, or networkit
anchor_searcher_resolver: Resolver[AnchorSearcher] = Resolver.from_subclasses(
    base=AnchorSearcher,
    default=CSGraphAnchorSearcher,
)


class AnchorTokenizer(Tokenizer):
    """
    Tokenize entities by representing them as a bag of anchor entities.

    The entities are chosen by shortest path distance.
    """

    def __init__(
        self,
        selection: HintOrType[AnchorSelection] = None,
        selection_kwargs: OptionalKwargs = None,
        searcher: HintOrType[AnchorSearcher] = None,
        searcher_kwargs: OptionalKwargs = None,
    ) -> None:
        """
        Initialize the tokenizer.

        :param selection:
            the anchor node selection strategy.
        :param selection_kwargs:
            additional keyword-based arguments passed to the selection strategy
        :param searcher:
            the component for searching the closest anchors for each entity
        :param searcher_kwargs:
            additional keyword-based arguments passed to the searcher
        """
        self.anchor_selection = anchor_selection_resolver.make(selection, pos_kwargs=selection_kwargs)
        self.searcher = anchor_searcher_resolver.make(searcher, pos_kwargs=searcher_kwargs)

    def __call__(
        self,
        mapped_triples: MappedTriples,
        num_tokens: int,
        num_entities: int,
        num_relations: int,
    ) -> torch.LongTensor:  # noqa: D102
        edge_index = mapped_triples[:, [0, 2]].numpy().T
        # select anchors
        anchors = self.anchor_selection(edge_index=edge_index)
        # find closest anchors
        tokens = self.searcher(edge_index=edge_index, anchors=anchors, k=num_tokens)
        # convert to torch
        return torch.as_tensor(tokens, dtype=torch.long)


tokenizer_resolver: Resolver[Tokenizer] = Resolver.from_subclasses(
    base=Tokenizer,
    default=RelationTokenizer,
)


def resolve_aggregation(
    aggregation: Union[None, str, Callable[[torch.FloatTensor, int], torch.FloatTensor]],
) -> Callable[[torch.FloatTensor, int], torch.FloatTensor]:
    """
    Resolve the aggregation function.

    .. warning ::
        This function does *not* check whether torch.<aggregation> is a method which is a valid aggregation.

    :param aggregation:
        the aggregation choice. Can be either
        1. None, in which case the torch.mean is returned
        2. a string, in which case torch.<aggregation> is returned
        3. a callable, which is returned without change

    :return:
        the chosen aggregation function.
    """
    if aggregation is None:
        return torch.mean

    if isinstance(aggregation, str):
        if aggregation not in AGGREGATIONS:
            logger.warning(
                f"aggregation={aggregation} is not one of the predefined ones ({sorted(AGGREGATIONS.keys())}).",
            )
        return getattr(torch, aggregation)

    return aggregation


class NodePieceRepresentation(RepresentationModule):
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
    tokens: RepresentationModule

    #: the entity-to-token mapping
    assignment: torch.LongTensor

    #: the padding idx, if any
    padding_idx: Optional[int]

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        token_representation: Union[EmbeddingSpecification, RepresentationModule],
        aggregation: Union[None, str, Callable[[torch.FloatTensor, int], torch.FloatTensor]] = None,
        num_tokens: int = 2,
        tokenizer: HintOrType[Tokenizer] = None,
        tokenizer_kwargs: OptionalKwargs = None,
        shape: Optional[Sequence[int]] = None,
    ):
        """
        Initialize the representation.

        :param triples_factory:
            the triples factory
        :param token_representation:
            the token representation specification, or pre-instantiated representation module. For the latter, the
            number of representations must be $2 * num_relations + 1$.
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
        :param num_tokens:
            the number of tokens for each entity.
        :param shape:
            the shape of an individual representation. Only necessary, if aggregation results in a change of dimensions.
        """
        mapped_triples = triples_factory.mapped_triples
        if triples_factory.create_inverse_triples:
            # inverse triples are created afterwards implicitly
            mapped_triples = mapped_triples[mapped_triples[:, 1] < triples_factory.real_num_relations]

        tokenizer_inst = tokenizer_resolver.make(tokenizer, pos_kwargs=tokenizer_kwargs)
        assignment = tokenizer_inst(
            mapped_triples=mapped_triples,
            num_tokens=num_tokens,
            num_entities=triples_factory.num_entities,
            num_relations=triples_factory.real_num_relations,
        )
        # fill padding
        padding = assignment < 0
        if padding.any():
            assignment[padding] = self.padding_idx = assignment.max().item() + 1
        else:
            self.padding_idx = None
        total_num_tokens = assignment.max().item() + 1

        # create token representations
        if isinstance(token_representation, EmbeddingSpecification):
            token_representation = token_representation.make(
                num_embeddings=total_num_tokens,
            )
        if token_representation.max_id != total_num_tokens:
            raise ValueError(
                f"If a pre-instantiated representation is provided, it has to have 2 * num_relations + 1= "
                f"{total_num_tokens} representations, but has {token_representation.max_id}",
            )

        # super init; has to happen *before* any parameter or buffer is assigned
        super().__init__(max_id=triples_factory.num_entities, shape=shape or token_representation.shape)

        # Assign default aggregation
        self.aggregation = resolve_aggregation(aggregation=aggregation)

        # assign module
        self.tokens = token_representation
        self.aggregation_index = -(1 + len(token_representation.shape))
        self.register_buffer(name="assignment", tensor=assignment)

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # get token IDs, shape: (*, k)
        token_ids = self.assignment
        if indices is not None:
            token_ids = token_ids[indices]

        # lookup token representations, shape: (*, k, d)
        x = self.tokens(token_ids)

        # aggregate
        x = self.aggregation(x, self.aggregation_index)

        return x
