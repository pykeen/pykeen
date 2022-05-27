# -*- coding: utf-8 -*-

"""Implementation of the R-GCN model."""

from typing import Any, Mapping, Optional

import torch
from class_resolver import Hint, HintOrType
from torch import nn

from ..nbase import ERModel
from ...nn.message_passing import Decomposition, RGCNRepresentation
from ...nn.modules import Interaction
from ...nn.representation import Representation
from ...nn.weighting import EdgeWeighting
from ...regularizers import Regularizer
from ...triples import CoreTriplesFactory
from ...typing import Initializer, RelationRepresentation

__all__ = [
    "RGCN",
]


class RGCN(
    ERModel[torch.FloatTensor, RelationRepresentation, torch.FloatTensor],
):
    r"""An implementation of R-GCN from [schlichtkrull2018]_.

    The Relational Graph Convolutional Network (R-GCN) comprises three parts:

    1. A GCN-based entity encoder that computes enriched representations for entities, cf.
       :class:`pykeen.nn.message_passing.RGCNRepresentations`. The representation for entity $i$ at level
       $l \in (1,\dots,L)$ is denoted as $\textbf{e}_i^l$.
       The GCN is modified to use different weights depending on the type of the relation.
    2. Relation representations $\textbf{R}_{r} \in \mathbb{R}^{d \times d}$ is a diagonal matrix that are learned
       independently from the GCN-based encoder.
    3. An arbitrary interaction model which computes the plausibility of facts given the enriched representations,
       cf. :class:`pykeen.nn.modules.Interaction`.

    Scores for each triple $(h,r,t) \in \mathcal{K}$ are calculated by using the representations in the final level
    of the GCN-based encoder $\textbf{e}_h^L$ and $\textbf{e}_t^L$ along with relation representation $\textbf{R}_{r}$.
    While the original implementation of R-GCN used the DistMult model and we use it as a default, this implementation
    allows the specification of an arbitrary interaction model.

    .. math::

        f(h,r,t) = \textbf{e}_h^L \textbf{R}_{r} \textbf{e}_t^L

    .. seealso::

       - `PyTorch Geometric's implementation of R-GCN
         <https://github.com/rusty1s/pytorch_geometric/blob/1.3.2/examples/rgcn.py>`_
       - `DGL's implementation of R-GCN
         <https://github.com/dmlc/dgl/tree/v0.4.0/examples/pytorch/rgcn>`_
    ---
    name: R-GCN
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
        edge_dropout=dict(type=float, low=0.0, high=0.9),
        self_loop_dropout=dict(type=float, low=0.0, high=0.9),
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
        relation_representations: HintOrType[Representation] = None,
        relation_initializer: Hint[Initializer] = nn.init.xavier_uniform_,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        interaction: HintOrType[Interaction[torch.FloatTensor, RelationRepresentation, torch.FloatTensor]] = "DistMult",
        interaction_kwargs: Optional[Mapping[str, Any]] = None,
        use_bias: bool = True,
        activation: Hint[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Hint[EdgeWeighting] = None,
        decomposition: Hint[Decomposition] = None,
        decomposition_kwargs: Optional[Mapping[str, Any]] = None,
        regularizer: Hint[Regularizer] = None,
        regularizer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the model.

        :param triples_factory:
            the (training) triples factory
        :param embedding_dim:
            the embedding dimension
        :param num_layers: >0
            the number of layers

        :param base_entity_initializer:
            the entity base representation initializer
        :param base_entity_initializer_kwargs:
            the entity base representation initializer's keyword-based parameters

        :param relation_representations:
            the relation representations, or a hint thereof
        :param relation_initializer:
            the entity base representation initializer
        :param relation_initializer_kwargs:
            the entity base representation initializer's keyword-based parameters

        :param interaction:
            the interaction function, or a hint thereof
        :param interaction_kwargs:
            additional keyword-based parameters passed to the interaction function

        :param use_bias:
            whether to use a bias on the message passing layers

        :param activation:
            the activation function, or a hint thereof
        :param activation_kwargs:
            additional keyword-based parameters passed to the activation function

        :param edge_dropout:
            the edge dropout, except for self-loops
        :param self_loop_dropout:
            the self-loop dropout
        :param edge_weighting:
            the edge weighting

        :param decomposition:
            the convolution weight decomposition
        :param decomposition_kwargs:
            additional keyword-based parameters passed to the weight decomposition
        :param regularizer:
            the regularizer applied to the base representations
        :param regularizer_kwargs:
            additional keyword-based parameters passed to the regularizer

        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        super().__init__(
            entity_representations=RGCNRepresentation,
            entity_representations_kwargs=dict(
                triples_factory=triples_factory,
                entity_representations_kwargs=dict(
                    shape=embedding_dim,
                    initializer=base_entity_initializer,
                    initializer_kwargs=base_entity_initializer_kwargs,
                ),
                num_layers=num_layers,
                use_bias=use_bias,
                activation=activation,
                activation_kwargs=activation_kwargs,
                edge_dropout=edge_dropout,
                self_loop_dropout=self_loop_dropout,
                edge_weighting=edge_weighting,
                decomposition=decomposition,
                decomposition_kwargs=decomposition_kwargs,
                # cf. https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/decoders/bilinear_diag.py#L64-L67  # noqa: E501
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            relation_representations=relation_representations,
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                initializer_kwargs=relation_initializer_kwargs,
                # cf. https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/decoders/bilinear_diag.py#L64-L67  # noqa: E501
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            triples_factory=triples_factory,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            **kwargs,
        )
