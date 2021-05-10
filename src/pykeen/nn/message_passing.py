# -*- coding: utf-8 -*-

"""Various decompositions for R-GCN."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
from class_resolver import Resolver
from torch import nn
from torch.nn import functional

__all__ = [
    "Decomposition",
    'BasesDecomposition',
    'BlockDecomposition',
    "decomposition_resolver",
]


def _reduce_relation_specific(
    relation: int,
    source: torch.LongTensor,
    target: torch.LongTensor,
    edge_type: torch.LongTensor,
    edge_weights: Optional[torch.FloatTensor],
) -> Union[Tuple[torch.LongTensor, torch.LongTensor, Optional[torch.FloatTensor]], Tuple[None, None, None]]:
    """Reduce edge information to one relation.

    :param relation:
        The relation ID.
    :param source: shape: (num_edges,)
        The source node IDs.
    :param target: shape: (num_edges,)
        The target node IDs.
    :param edge_type: shape: (num_edges,)
        The edge types.
    :param edge_weights: shape: (num_edges,)
        The edge weights.

    :return:
        The source, target, weights for edges related to the desired relation type.
    """
    # mask, shape: (num_edges,)
    edge_mask = edge_type == relation
    if not edge_mask.any():
        return None, None, None

    source_r = source[edge_mask]
    target_r = target[edge_mask]
    if edge_weights is not None:
        edge_weights = edge_weights[edge_mask]

    # bi-directional message passing
    source_r, target_r = torch.cat([source_r, target_r]), torch.cat([target_r, source_r])
    if edge_weights is not None:
        edge_weights = torch.cat([edge_weights, edge_weights])

    return source_r, target_r, edge_weights


class Decomposition(nn.Module, ABC):
    """Base module for relation-specific message passing."""

    def __init__(
        self,
        input_dim: int,
        num_relations: int,
        output_dim: Optional[int] = None,
    ):
        """Initialize the layer.

        :param input_dim: >0
            The input dimension.
        :param num_relations: >0
            The number of relations.
        :param output_dim: >0
            The output dimension. If None is given, defaults to input_dim.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_relations = num_relations
        if output_dim is None:
            output_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(
        self,
        x: torch.FloatTensor,
        node_keep_mask: Optional[torch.BoolTensor],
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Relation-specific message passing from source to target.

        :param x: shape: (num_nodes, input_dim)
            The node representations.
        :param node_keep_mask: shape: (num_nodes,)
            The node-keep mask for self-loop dropout.
        :param source: shape: (num_edges,)
            The source indices.
        :param target: shape: (num_edges,)
            The target indices.
        :param edge_type: shape: (num_edges,)
            The edge types.
        :param edge_weights: shape: (num_edges,)
            Precomputed edge weights.

        :return: shape: (num_nodes, output_dim)
            The enriched node embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self):
        """Reset the parameters of this layer."""
        raise NotImplementedError


class BasesDecomposition(Decomposition):
    """Represent relation-weights as a linear combination of base transformation matrices."""

    def __init__(
        self,
        input_dim: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        output_dim: Optional[int] = None,
        memory_intense: bool = False,
    ):
        """Initialize the layer.

        :param input_dim: >0
            The input dimension.
        :param num_relations: >0
            The number of relations.
        :param num_bases: >0
            The number of bases to use.
        :param output_dim: >0
            The output dimension. If None is given, defaults to input_dim.
        :param memory_intense:
            Enable memory-intense forward pass which may be faster, in particular if the number of different relations
            is small.
        """
        super().__init__(
            input_dim=input_dim,
            num_relations=num_relations,
            output_dim=output_dim,
        )

        # Heuristic for default value
        if num_bases is None:
            logging.info('No num_bases was provided. Falling back to 2.')
            num_bases = 2

        if num_bases > num_relations:
            raise ValueError('The number of bases should not exceed the number of relations.')

        # weights
        self.bases = nn.Parameter(
            torch.empty(
                num_bases,
                self.input_dim,
                self.output_dim,
            ), requires_grad=True)
        self.relation_base_weights = nn.Parameter(
            torch.empty(
                num_relations + 1,
                num_bases,
            ), requires_grad=True)

        self.memory_intense = memory_intense

    def reset_parameters(self):  # noqa: D102
        nn.init.xavier_normal_(self.bases)
        # Random convex-combination of bases for initialization (guarantees that initial weight matrices are
        # initialized properly)
        # We have one additional relation for self-loops
        nn.init.uniform_(self.relation_base_weights)
        functional.normalize(self.relation_base_weights.data, p=1, dim=1, out=self.relation_base_weights.data)

    def _get_weight(self, relation_id: int) -> torch.FloatTensor:
        """Construct weight matrix for a specific relation ID.

        :param relation_id:
            The relation ID.

        :return:
            A 2-D matrix.
        """
        return torch.einsum('bij,b->ij', self.bases, self.relation_base_weights[relation_id])

    def _forward_memory_intense(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        out: torch.FloatTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # other relations
        m = torch.einsum(
            'mi,mb,bij->mj',
            x.index_select(dim=0, index=source),
            self.relation_base_weights.index_select(dim=0, index=edge_type),
            self.bases,
        )
        if edge_weights is not None:
            m = m * edge_weights.unsqueeze(dim=-1)
        return out.index_add(dim=0, index=target, source=m)

    def _forward_memory_light(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        out: torch.FloatTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # other relations
        for r in range(self.num_relations):
            # Select source and target indices as well as edge weights for the
            # currently considered relation
            source_r, target_r, weights_r = _reduce_relation_specific(
                relation=r,
                source=source,
                target=target,
                edge_type=edge_type,
                edge_weights=edge_weights,
            )

            # skip relations without edges
            if source_r is None:
                continue

            # compute message, shape: (num_edges_of_type, output_dim)
            w = self._get_weight(relation_id=r)
            # since we may have one node ID appearing multiple times as source
            # ID, we can save some computation by first reducing to the unique
            # source IDs, compute transformed representations and afterwards
            # select these representations for the correct edges.
            uniq_source_r, inv_source_r = source_r.unique(return_inverse=True)
            # select unique source node representations
            m = x[uniq_source_r]
            # transform representations by relation specific weight
            m = m @ w
            # select the uniquely transformed representations for each edge
            m = m.index_select(dim=0, index=inv_source_r)

            # optional message weighting
            if weights_r is not None:
                m = m * weights_r.unsqueeze(dim=-1)

            # message aggregation
            out = out.index_add(dim=0, index=target_r, source=m)

        return out

    def forward(
        self,
        x: torch.FloatTensor,
        node_keep_mask: Optional[torch.BoolTensor],
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # self-loops first
        # the last relation_id refers to the self-loop
        w = self._get_weight(relation_id=self.num_relations)
        if node_keep_mask is not None:
            assert node_keep_mask.shape == x.shape[:1]
            out = torch.empty_like(x)
            out[node_keep_mask] = x[node_keep_mask] @ w
            out[~node_keep_mask] = 0.0
        else:
            out = x @ w

        if self.memory_intense:
            _forward = self._forward_memory_intense
        else:
            _forward = self._forward_memory_light

        return _forward(
            x=x,
            source=source,
            target=target,
            edge_type=edge_type,
            out=out,
            edge_weights=edge_weights,
        )


class BlockDecomposition(Decomposition):
    """Represent relation-specific weight matrices via block-diagonal matrices."""

    def __init__(
        self,
        input_dim: int,
        num_relations: int,
        num_blocks: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        """Initialize the layer.

        :param input_dim: >0
            The input dimension.
        :param num_relations: >0
            The number of relations.
        :param num_blocks: >0
            The number of blocks to use. Has to be a divisor of input_dim.
        :param output_dim: >0
            The output dimension. If None is given, defaults to input_dim.
        """
        super().__init__(
            input_dim=input_dim,
            num_relations=num_relations,
            output_dim=output_dim,
        )

        if num_blocks is None:
            logging.info('Using a heuristic to determine the number of blocks.')
            num_blocks = min(i for i in range(2, input_dim + 1) if input_dim % i == 0)

        block_size, remainder = divmod(input_dim, num_blocks)
        if remainder != 0:
            raise NotImplementedError(
                'With block decomposition, the embedding dimension has to be divisible by the number of'
                f' blocks, but {input_dim} % {num_blocks} != 0.',
            )

        self.blocks = nn.Parameter(
            data=torch.empty(
                num_relations + 1,
                num_blocks,
                block_size,
                block_size,
            ), requires_grad=True)
        self.num_blocks = num_blocks
        self.block_size = block_size

    def reset_parameters(self):  # noqa: D102
        block_size = self.blocks.shape[-1]
        # Xavier Glorot initialization of each block
        std = torch.sqrt(torch.as_tensor(2.)) / (2 * block_size)
        nn.init.normal_(self.blocks, std=std)

    def forward(
        self,
        x: torch.FloatTensor,
        node_keep_mask: Optional[torch.BoolTensor],
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # view as blocks
        x = x.view(-1, self.num_blocks, self.block_size)

        # self-loop first
        out = torch.zeros_like(x)
        w = self.blocks[-1]
        if node_keep_mask is not None:
            out[node_keep_mask] = torch.einsum('nbi,bij->nbj', x[node_keep_mask], w)
        else:
            out = torch.einsum('nbi,bij->nbj', x, w)

        # other relations
        for r in range(self.num_relations):
            source_r, target_r, weights_r = _reduce_relation_specific(
                relation=r,
                source=source,
                target=target,
                edge_type=edge_type,
                edge_weights=edge_weights,
            )

            # skip relations without edges
            if source_r is None:
                continue

            # compute message, shape: (num_edges_of_type, num_blocks, block_size)
            uniq_source_r, inv_source_r = source_r.unique(return_inverse=True)
            w_r = self.blocks[r]
            m = torch.einsum('nbi,bij->nbj', x[uniq_source_r], w_r).index_select(dim=0, index=inv_source_r)

            # optional message weighting
            if weights_r is not None:
                m = m * weights_r.unsqueeze(dim=1).unsqueeze(dim=2)

            # message aggregation
            out.index_add_(dim=0, index=target_r, source=m)

        return out.reshape(-1, self.output_dim)


decomposition_resolver = Resolver.from_subclasses(base=Decomposition, default=BasesDecomposition)
