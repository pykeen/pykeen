# -*- coding: utf-8 -*-

"""Implementation of the R-GCN model."""

from typing import Any, Mapping, Optional

import torch
from class_resolver import Hint
from torch import nn

from ..nbase import ERModel, EmbeddingSpecificationHint
from ...nn.emb import EmbeddingSpecification, RGCNRepresentations
from ...nn.message_passing import Decomposition
from ...nn.modules import Interaction, interaction_resolver
from ...nn.weighting import EdgeWeighting
from ...triples import CoreTriplesFactory
from ...typing import Initializer, RelationRepresentation

__all__ = [
    "RGCN",
]


class RGCN(
    ERModel[torch.FloatTensor, RelationRepresentation, torch.FloatTensor],
):
    r"""An implementation of R-GCN from [schlichtkrull2018]_.

    The Relational Graph Convolutional Network (R-GCN) comprises two parts:

    1. A GCN-based entity encoder that computes enriched representations for entities, cf.
       :class:`pykeen.nn.emb.RGCNRepresentations`. The GCN is modified to use different weights depending on the
       type of the relation.
    2. An arbitrary interaction model which computes the plausibility of facts given the enriched representations, 
       cf. :class:`pykeen.nn.modules.Interaction`.

    .. todo:: Move the following part to :class:`pykeen.nn.emb.RGCNRepresentations`

    The GCN employed by the entity encoder is adapted to include typed edges.
    The forward pass of the GCN is defined by:

     .. math::

        \textbf{e}_{i}^{l+1} = \sigma \left( \sum_{r \in \mathcal{R}}\sum_{j\in \mathcal{N}_{i}^{r}}
        \frac{1}{c_{i,r}} \textbf{W}_{r}^{l} \textbf{e}_{j}^{l} + \textbf{W}_{0}^{l} \textbf{e}_{i}^{l}\right)

    where $\mathcal{N}_{i}^{r}$ is the set of neighbors of node $i$ that are connected to
    $i$ by relation $r$, $c_{i,r}$ is a fixed normalization constant (but it can also be introduced as an additional
    parameter), and $\textbf{W}_{r}^{l} \in \mathbb{R}^{d^{(l)} \times d^{(l)}}$ and
    $\textbf{W}_{0}^{l} \in \mathbb{R}^{d^{(l)} \times d^{(l)}}$ are weight matrices of the `l`-th layer of the
    R-GCN.

    The encoder aggregates for each node $e_i$ the latent representations of its neighbors and its
    own latent representation $e_{i}^{l}$ into a new latent representation $e_{i}^{l+1}$.
    In contrast to standard GCN, R-GCN defines relation specific transformations
    $\textbf{W}_{r}^{l}$ which depend on the type and direction of an edge.

    The interaction model computes the plausibility score given the node representations $\textbf{e}_{i}^{L}$ that are
    computed by the last layer $L$ of the R-GCN, i.e., for a given triple $(h,r,t) \in \mathcal{K}$, the
    corresponding node representations $h:=e_i^L$ and $t:=e_j^L$ are used:

    .. math::

        f(h,r,t) = \textbf{h} \textbf{R}_{r} \textbf{t}

    where $\textbf{R}_{r} \in \mathbb{R}^{d \times d}$ is a diagonal matrix and $f(h,r,t)$ is the
    interaction model of DistMult (DistMult was employed in the original work, however, the general approach is not
    restricted to DistMult).

    The :class:`pykeen.nn.message_passing.Decomposition` module provides an interface for a regularization approach
    that reduces the number of parameters required for the relation-specific transformation matrices and mitigates
    over-fitting. The two approaches published with R-GCN are implemented in PyKEEN. The first, basis decomposition
    (:class:`pykeen.nn.message_passing.BasesDecomposition`), represents the relation-specific transformation matrices
    as a weighted combination of base matrices, $\{\mathbf{B}_i^l\}_{i=1}^{B}$, i.e.,

    .. math::

        \mathbf{W}_r^l = \sum \limits_{b=1}^B \alpha_{rb} \mathbf{B}^l_i

    The second, block-diagonal decomposition (:class:`pykeen.nn.message_passing.BlockDecomposition`),
    restricts each transformation matrix to a block-diagonal-matrix, i.e.,

    .. math::

        \mathbf{W}_r^l = diag(\mathbf{B}_{r,1}^l, \ldots, \mathbf{B}_{r,B}^l)

    where $\mathbf{B}_{r,i} \in \mathbb{R}^{(d^{(l) }/ B) \times (d^{(l)} / B)}$.

    .. seealso::

       - `Pytorch Geometric's implementation of R-GCN
         <https://github.com/rusty1s/pytorch_geometric/blob/1.3.2/examples/rgcn.py>`_
       - `DGL's implementation of R-GCN
         <https://github.com/dmlc/dgl/tree/v0.4.0/examples/pytorch/rgcn>`_
    ---
    citation:
        author: Schlichtkrull
        year: 2018
        link: https://arxiv.org/pdf/1703.06103
    """

    #: The default strategy for optimizing the model"s hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=32, high=512, q=32),
        num_layers=dict(type=int, low=1, high=5, q=1),
        use_bias=dict(type="bool"),
        use_batch_norm=dict(type="bool"),
        activation_cls=dict(type="categorical", choices=[nn.ReLU, nn.LeakyReLU]),
        interaction=dict(type="categorical", choices=["distmult", "complex", "ermlp"]),
        edge_dropout=dict(type=float, low=0.0, high=.9),
        self_loop_dropout=dict(type=float, low=0.0, high=.9),
        edge_weighting=dict(type="categorical", choices=["inverse_in_degree", "inverse_out_degree", "symmetric"]),
        decomposition=dict(type="categorical", choices=["bases", "blocks"]),
        # TODO: Decomposition kwargs
        # num_bases=dict(type=int, low=2, high=100, q=1),
        # num_blocks=dict(type=int, low=2, high=20, q=1),
    )

    def __init__(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        embedding_dim: int = 500,
        num_layers: int = 2,
        # https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/encoders/affine_transform.py#L24-L28
        base_entity_initializer: Hint[Initializer] = nn.init.xavier_uniform_,
        base_entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = nn.init.xavier_uniform_,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_representations: EmbeddingSpecificationHint = None,
        interaction: Interaction[torch.FloatTensor, RelationRepresentation, torch.FloatTensor],
        interaction_kwargs: Optional[Mapping[str, Any]] = None,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        activation: Hint[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Hint[EdgeWeighting] = None,
        decomposition: Hint[Decomposition] = None,
        decomposition_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        # create enriched entity representations
        entity_representations = RGCNRepresentations(
            triples_factory=triples_factory,
            embedding_specification=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=base_entity_initializer,
                initializer_kwargs=base_entity_initializer_kwargs,
            ),
            num_layers=num_layers,
            use_bias=use_bias,
            use_batch_norm=use_batch_norm,
            activation=activation,
            activation_kwargs=activation_kwargs,
            edge_dropout=edge_dropout,
            self_loop_dropout=self_loop_dropout,
            edge_weighting=edge_weighting,
            decomposition=decomposition,
            decomposition_kwargs=decomposition_kwargs,
        )

        # Resolve interaction function
        interaction = interaction_resolver.make(query=interaction, pos_kwargs=interaction_kwargs)

        # set default relation representation
        if relation_representations is None:
            relation_representations = EmbeddingSpecification(
                shape=entity_representations.shape,
                initializer=relation_initializer,
                initializer_kwargs=relation_initializer_kwargs,
            )
        super().__init__(
            entity_representations=entity_representations,
            relation_representations=relation_representations,
            triples_factory=triples_factory,
            interaction=interaction,
            **kwargs,
        )
