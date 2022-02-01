"""Node Piece representations."""
import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy
import numpy.linalg
import scipy.sparse
import scipy.sparse.csgraph
import torch
from class_resolver import HintOrType, OptionalKwargs, Resolver

from .emb import EmbeddingSpecification, RepresentationModule
from ..constants import AGGREGATIONS
from ..triples import CoreTriplesFactory
from ..triples.splitting import get_absolute_split_sizes, normalize_ratios
from ..typing import MappedTriples, OneOrSequence
from ..utils import format_relative_comparison

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
    ) -> Tuple[int, torch.LongTensor]:
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
    ) -> Tuple[int, torch.LongTensor]:  # noqa: D102
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

        return 2 * num_relations + 1, assignment


def _edge_index_to_sparse_matrix(
    edge_index: numpy.ndarray,
    num_entities: Optional[int] = None,
) -> scipy.sparse.spmatrix:
    if num_entities is None:
        num_entities = edge_index.max().item() + 1
    return scipy.sparse.coo_matrix(
        (
            numpy.ones_like(edge_index[0], dtype=bool),
            tuple(edge_index),
        ),
        shape=(num_entities, num_entities),
    )


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

    def extra_repr(self) -> Iterable[str]:
        """Extra components for __repr__."""
        yield f"num_anchors={self.num_anchors}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self.extra_repr())})"


class DegreeAnchorSelection(AnchorSelection):
    """Select entities according to their (undirected) degree."""

    def __call__(self, edge_index: numpy.ndarray) -> numpy.ndarray:  # noqa: D102
        unique, counts = numpy.unique(edge_index, return_counts=True)
        top_ids = numpy.argpartition(counts, max(counts.size - self.num_anchors, 0))[-self.num_anchors :]
        return unique[top_ids]


class PageRankAnchorSelection(AnchorSelection):
    """Select entities according to their page rank."""

    def __init__(
        self,
        num_anchors: int = 32,
        max_iter: int = 1_000,
        alpha: float = 0.05,
        epsilon: float = 1.0e-04,
    ) -> None:
        """
        Initialize the selection strategy.

        :param num_anchors:
            the number of anchors to select
        :param max_iter:
            the maximum number of power iterations
        :param alpha:
            the smoothing value / teleport probability
        :param epsilon:
            a constant to check for convergence
        """
        super().__init__(num_anchors=num_anchors)
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.epsilon = epsilon

    def extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().extra_repr()
        yield f"max_iter={self.max_iter}"
        yield f"alpha={self.alpha}"
        yield f"epsilon={self.epsilon}"

    def __call__(self, edge_index: numpy.ndarray) -> numpy.ndarray:  # noqa: D102
        # convert to sparse matrix
        adj = _edge_index_to_sparse_matrix(edge_index=edge_index)
        # symmetrize
        # TODO: should we add self-links
        # adj = (adj + adj.transpose() + scipy.sparse.eye(m=adj.shape[0], format="coo")).tocsr()
        adj = (adj + adj.transpose()).tocsr()
        # degree
        degree_inv = numpy.reciprocal(numpy.asarray(adj.sum(axis=0)))[0]
        n = degree_inv.shape[0]
        # power iteration
        x = numpy.full(shape=(n,), fill_value=1.0 / n)
        x_old = x
        no_convergence = True
        for i in range(self.max_iter):
            x = self.beta * adj.dot(degree_inv * x) + self.alpha
            if numpy.linalg.norm(x - x_old, ord=float("+inf")) < self.epsilon:
                logger.debug(f"Converged after {i} iterations up to {self.epsilon}.")
                no_convergence = False
                break
            x_old = x
        if no_convergence:
            logger.warning(f"No covergence after {self.max_iter} iterations with epsilon={self.epsilon}.")
        return numpy.argpartition(x, max(x.size - self.num_anchors, 0))[-self.num_anchors :]


class MixtureAnchorSelection(AnchorSelection):
    """A weighted mixture of different anchor selection strategies."""

    def __init__(
        self,
        selections: Sequence[HintOrType[AnchorSelection]],
        ratios: Union[None, float, Sequence[float]] = None,
        selections_kwargs: OneOrSequence[OptionalKwargs] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the selection strategy.

        :param selections:
            the individual selections
        :param ratios:
            the ratios, cf. normalize_ratios. None means uniform ratios
        :param selection_kwargs:
            additional keyword-based arguments for the individual selection strategies
        :param kwargs:
            additional keyword-based arguments passed to AnchorSelection.__init__,
            in particular, the total number of anchors.
        """
        super().__init__(**kwargs)
        n_selections = len(selections)
        # input normalization
        if selections_kwargs is None:
            selections_kwargs = [None] * n_selections
        if ratios is None:
            ratios = numpy.ones(shape=(n_selections,)) / n_selections
        # determine absolute number of anchors for each strategy
        num_anchors = get_absolute_split_sizes(n_total=self.num_anchors, ratios=normalize_ratios(ratios=ratios))
        self.selections = [
            anchor_selection_resolver.make(selection, selection_kwargs, num_anchors=num)
            for selection, selection_kwargs, num in zip(selections, selections_kwargs, num_anchors)
        ]
        # if pre-instantiated
        for selection, num in zip(self.selections, num_anchors):
            if selection.num_anchors != num:
                logger.warning(f"{selection} had wrong number of anchors. Setting to {num}")
                selection.num_anchors = num

    def extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().extra_repr()
        yield f"selections={self.selections}"

    def __call__(self, edge_index: numpy.ndarray) -> numpy.ndarray:  # noqa: D102
        return numpy.concatenate([selection(edge_index=edge_index) for selection in self.selections])


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

    def extra_repr(self) -> Iterable[str]:
        """Extra components for __repr__."""
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self.extra_repr())})"


class CSGraphAnchorSearcher(AnchorSearcher):
    """Find closest anchors using scipy.sparse.csgraph."""

    def __call__(self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int) -> numpy.ndarray:  # noqa: D102
        # convert to adjacency matrix
        adjacency = _edge_index_to_sparse_matrix(edge_index=edge_index).tocsr()
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
            numpy.argpartition(distances, kth=min(k, distances.shape[0]), axis=0)[:k, :].T,
            dtype=torch.long,
        )


class ScipySparseAnchorSearcher(AnchorSearcher):
    """Find closest anchors using scipy.sparse."""

    def __init__(self, max_iter: int = 5) -> None:
        """
        Initialize the searcher.

        :param max_iter:
            the maximum number of hops to consider
        """
        self.max_iter = max_iter

    def extra_repr(self) -> Iterable[str]:  # noqa: D102
        yield from super().extra_repr()
        yield f"max_iter={self.max_iter}"

    @staticmethod
    def create_adjacency(
        edge_index: numpy.ndarray,
    ) -> scipy.sparse.spmatrix:
        """
        Create a sparse adjacency matrix from a given edge index.

        :param edge_index: shape: (2, m)
            the edge index

        :return: shape: (n, n)
            a square sparse adjacency matrix
        """
        # infer shape
        num_entities = edge_index.max().item() + 1
        # create adjacency matrix
        adjacency = scipy.sparse.coo_matrix(
            (
                numpy.ones_like(edge_index[0], dtype=bool),
                tuple(edge_index),
            ),
            shape=(num_entities, num_entities),
            dtype=bool,
        )
        # symmetric + self-loops
        adjacency = adjacency + adjacency.transpose() + scipy.sparse.eye(num_entities, dtype=bool, format="coo")
        adjacency = adjacency.tocsr()
        logger.debug(
            f"Created sparse adjacency matrix of shape {adjacency.shape} where "
            f"{format_relative_comparison(part=adjacency.nnz, total=numpy.prod(adjacency.shape))} "
            f"are non-zero entries.",
        )
        return adjacency

    @staticmethod
    def bfs(
        anchors: numpy.ndarray,
        adjacency: scipy.sparse.spmatrix,
        max_iter: int,
        k: int,
    ) -> numpy.ndarray:
        """
        Determine the candidate pool using breadth-first search.

        :param anchors: shape: (a,)
            the anchor node IDs
        :param adjacency: shape: (n, n)
            the adjacency matrix
        :param max_iter:
            the maximum number of hops to consider
        :param k:
            the minimum number of anchor nodes to reach

        :return: shape: (n, a)
            a boolean array indicating whether anchor $j$ is in the set of $k$ closest anchors for node $i$
        """
        num_entities = adjacency.shape[0]
        # for each entity, determine anchor pool by BFS
        num_anchors = len(anchors)

        # an array storing whether node i is reachable by anchor j
        reachable = numpy.zeros(shape=(num_entities, num_anchors), dtype=bool)
        reachable[anchors] = numpy.eye(num_anchors, dtype=bool)

        # an array indicating whether a node is closed, i.e., has found at least $k$ anchors
        final = numpy.zeros(shape=(num_entities,), dtype=bool)

        # the output
        pool = numpy.zeros(shape=(num_entities, num_anchors), dtype=bool)

        # TODO: take all (q-1) hop neighbors before selecting from q-hop
        old_reachable = reachable
        for i in range(max_iter):
            # propagate one hop
            reachable = adjacency.dot(reachable)
            # convergence check
            if (reachable == old_reachable).all():
                logger.warning(f"Search converged after iteration {i} without all nodes being reachable.")
                break
            old_reachable = reachable
            # copy pool if we have seen enough anchors and have not yet stopped
            num_reachable = reachable.sum(axis=1)
            enough = num_reachable >= k
            mask = enough & ~final
            logger.debug(
                f"Iteration {i}: {format_relative_comparison(enough.sum(), total=num_entities)} closed nodes.",
            )
            pool[mask] = reachable[mask]
            # stop once we have enough
            final |= enough
            if final.all():
                break
        return pool

    @staticmethod
    def select(
        pool: numpy.ndarray,
        k: int,
    ) -> numpy.ndarray:
        """
        Select $k$ anchors from the given pools.

        :param pool: shape: (n, a)
            the anchor candidates for each node (a binary array)
        :param k:
            the number of candidates to select

        :return: shape: (n, k)
            the selected anchors. May contain -1 if there is an insufficient number of  candidates
        """
        tokens = numpy.full(shape=(pool.shape[0], k), fill_value=-1, dtype=int)
        generator = numpy.random.default_rng()
        # TODO: can we replace this loop with something vectorized?
        for i, row in enumerate(pool):
            (this_pool,) = row.nonzero()
            chosen = generator.choice(a=this_pool, size=min(k, this_pool.size), replace=False, shuffle=False)
            tokens[i, : len(chosen)] = chosen
        return tokens

    def __call__(self, edge_index: numpy.ndarray, anchors: numpy.ndarray, k: int) -> numpy.ndarray:  # noqa: D102
        adjacency = self.create_adjacency(edge_index=edge_index)
        pool = self.bfs(anchors=anchors, adjacency=adjacency, max_iter=self.max_iter, k=k)
        return self.select(pool=pool, k=k)


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
        logger.info(f"Selecting anchors according to {self.anchor_selection}")
        anchors = self.anchor_selection(edge_index=edge_index)
        # find closest anchors
        logger.info(f"Searching closest anchors with {self.searcher}")
        tokens = self.searcher(edge_index=edge_index, anchors=anchors, k=num_tokens)
        num_empty = (tokens < 0).all(axis=1).sum()
        if num_empty > 0:
            logger.warning(
                f"{format_relative_comparison(part=num_empty, total=num_entities)} " f"do not have any anchor.",
            )
        # convert to torch
        return len(anchors) + 1, torch.as_tensor(tokens, dtype=torch.long)


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
        :param tokenizer:
            the tokenizer to use, cf. `pykeen.nn.node_piece.tokenizer_resolver`.
            TODO: support for using both Anchor and Relation tokenizers
        :param tokenizer_kwargs:
            additional keyword-based parameters passed to the tokenizer upon construction.
        :param shape:
            the shape of an individual representation. Only necessary, if aggregation results in a change of dimensions.
        """
        mapped_triples = triples_factory.mapped_triples
        if triples_factory.create_inverse_triples:
            # inverse triples are created afterwards implicitly
            mapped_triples = mapped_triples[mapped_triples[:, 1] < triples_factory.real_num_relations]

        tokenizer_inst = tokenizer_resolver.make(tokenizer, pos_kwargs=tokenizer_kwargs)
        total_num_tokens, assignment = tokenizer_inst(
            mapped_triples=mapped_triples,
            num_tokens=num_tokens,
            num_entities=triples_factory.num_entities,
            num_relations=triples_factory.real_num_relations,
        )
        # fill padding (nn.Embedding cannot deal with negative indices)
        padding = assignment < 0
        assignment[padding] = self.padding_idx = assignment.max().item() + 1

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
