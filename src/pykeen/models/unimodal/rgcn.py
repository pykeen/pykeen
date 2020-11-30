# -*- coding: utf-8 -*-

"""Implementation of the R-GCN model."""

from typing import Any, Callable, Mapping, Optional, Type

import torch
from torch import nn

from . import ComplEx, DistMult, ERMLP
from ..base import ERModel
from ...losses import Loss
from ...nn import EmbeddingSpecification, Interaction
from ...nn.modules import DistMultInteraction
from ...nn.representation import (
    RGCNRepresentations, inverse_indegree_edge_weights, inverse_outdegree_edge_weights,
    symmetric_edge_weights,
)
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'RGCN',
]


class RGCN(ERModel):
    """An implementation of R-GCN from [schlichtkrull2018]_.

    This model uses graph convolutions with relation-specific weights.

    .. seealso::

       - `Pytorch Geometric's implementation of R-GCN
         <https://github.com/rusty1s/pytorch_geometric/blob/1.3.2/examples/rgcn.py>`_
       - `DGL's implementation of R-GCN
         <https://github.com/dmlc/dgl/tree/v0.4.0/examples/pytorch/rgcn>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=1000, q=50),
        num_bases_or_blocks=dict(type=int, low=2, high=20, q=1),
        num_layers=dict(type=int, low=1, high=5, q=1),
        use_bias=dict(type='bool'),
        use_batch_norm=dict(type='bool'),
        activation_cls=dict(type='categorical', choices=[None, nn.ReLU, nn.LeakyReLU]),
        base_model_cls=dict(type='categorical', choices=[DistMult, ComplEx, ERMLP]),
        edge_dropout=dict(type=float, low=0.0, high=.9),
        self_loop_dropout=dict(type=float, low=0.0, high=.9),
        edge_weighting=dict(type='categorical', choices=[
            None,
            inverse_indegree_edge_weights,
            inverse_outdegree_edge_weights,
            symmetric_edge_weights,
        ]),
        decomposition=dict(type='categorical', choices=['basis', 'block']),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        interaction: Optional[Interaction] = None,
        embedding_dim: int = 500,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        num_bases_or_blocks: int = 5,
        num_layers: int = 2,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        activation_cls: Optional[Type[nn.Module]] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        sparse_messages_slcwa: bool = True,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Callable[
            [torch.LongTensor, torch.LongTensor],
            torch.FloatTensor,
        ] = inverse_indegree_edge_weights,
        decomposition: str = 'basis',
        buffer_messages: bool = True,
    ):
        if triples_factory.create_inverse_triples:
            raise ValueError('R-GCN handles edges in an undirected manner.')
        if interaction is None:
            interaction = DistMultInteraction()

        entity_representations = RGCNRepresentations(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            num_bases_or_blocks=num_bases_or_blocks,
            num_layers=num_layers,
            use_bias=use_bias,
            use_batch_norm=use_batch_norm,
            activation_cls=activation_cls,
            activation_kwargs=activation_kwargs,
            sparse_messages_slcwa=sparse_messages_slcwa,
            edge_dropout=edge_dropout,
            self_loop_dropout=self_loop_dropout,
            edge_weighting=edge_weighting,
            decomposition=decomposition,
            buffer_messages=buffer_messages,
            base_representations=None,
        )
        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
            interaction=interaction,
            entity_representations=entity_representations,
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
            ),
        )
