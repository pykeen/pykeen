# -*- coding: utf-8 -*-

"""Implementation of the R-GCN model."""

import logging
from os import path
from typing import Any, Mapping, Optional, Sequence, Type, Union

import torch
from torch import nn

from .decompositions import (
    BasesDecomposition, BlockDecomposition, RelationSpecificMessagePassing,
    decompositions, get_decomposition_cls,
)
from .weightings import EdgeWeighting, edge_weightings, get_edge_weighting
from .. import ComplEx, DistMult, ERMLP
from ... import Model
from ....losses import Loss
from ....nn import Embedding, RepresentationModule
from ....triples import TriplesFactory

__all__ = [
    'RGCN',
]

logger = logging.getLogger(name=path.basename(__file__))


class Bias(nn.Module):
    """A module wrapper for adding a bias."""

    def __init__(self, dim: int):
        """Initialize the module.

        :param dim: >0
            The dimension of the input.
        """
        super().__init__()
        self.bias = nn.Parameter(torch.empty(dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer's parameters."""
        nn.init.zeros_(self.bias)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Add the learned bias to the input.

        :param x: shape: (n, d)
            The input.

        :return:
            x + b[None, :]
        """
        return x + self.bias.unsqueeze(dim=0)


class RGCNRepresentations(RepresentationModule):
    """Entity representations enriched by R-GCN."""

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 500,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        num_layers: int = 2,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        activation_cls: Optional[Type[nn.Module]] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        sparse_messages_slcwa: bool = True,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Union[None, str, EdgeWeighting] = 'inverse_indegree',
        _decomposition: Union[None, str, Type[RelationSpecificMessagePassing]] = None,
        buffer_messages: bool = True,
        memory_intense: bool = False,
        base_representations: Optional[RepresentationModule] = None,
    ):
        super().__init__()

        # normalize representations
        if base_representations is None:
            base_representations = Embedding(
                num_embeddings=triples_factory.num_entities,
                embedding_dim=embedding_dim,
                # https://github.com/MichSchli/RelationPrediction/blob/c77b094fe5c17685ed138dae9ae49b304e0d8d89/code/encoders/affine_transform.py#L24-L28
                initializer=nn.init.xavier_uniform_,
            )
        self.base_embeddings = base_representations

        # buffering of messages
        self.buffer_messages = buffer_messages
        self.enriched_embeddings = None

        # if edge weighting is None, keep it as none. otherwise look it up
        self.edge_weighting = edge_weighting and get_edge_weighting(edge_weighting)
        self.edge_dropout = edge_dropout
        if self_loop_dropout is None:
            self_loop_dropout = edge_dropout
        self.self_loop_dropout = self_loop_dropout
        self.use_batch_norm = use_batch_norm
        if activation_cls is None:
            activation_cls = nn.ReLU
        self.activation_cls = activation_cls
        self.activation_kwargs = activation_kwargs
        if use_batch_norm:
            if use_bias:
                logger.warning('Disabling bias because batch normalization was used.')
            use_bias = False
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.sparse_messages_slcwa = sparse_messages_slcwa

        # Save graph using buffers, such that the tensors are moved together with the model
        h, r, t = self.triples_factory.mapped_triples.t()
        self.register_buffer('sources', h)
        self.register_buffer('targets', t)
        self.register_buffer('edge_types', r)

        decomposition_cls: Type[RelationSpecificMessagePassing] = get_decomposition_cls(_decomposition)

        message_passing_kwargs = {}
        if decomposition_cls is BasesDecomposition:
            message_passing_kwargs['num_bases'] = num_bases
            message_passing_kwargs['memory_intense'] = memory_intense
        elif decomposition_cls is BlockDecomposition:
            message_passing_kwargs['num_blocks'] = num_blocks

        layers = []
        for _ in range(num_layers):
            layers.append(decomposition_cls(
                input_dim=self.embedding_dim,
                num_relations=self.num_relations,
                **message_passing_kwargs,
            ))
            if self.use_bias:
                layers.append(Bias(dim=self.embedding_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(num_features=self.embedding_dim))
            layers.append(self.activation_cls(**(self.activation_kwargs or {})))
        self.layers = nn.ModuleList(layers)

    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()

        # invalidate enriched embeddings
        self.enriched_embeddings = None

    def reset_parameters(self):
        self.base_embeddings.reset_parameters()

        for m in self.layers:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            elif any(p.requires_grad for p in m.parameters()):
                logger.warning('Layers %s has parameters, but no reset_parameters.', m)

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Enrich the entity embeddings of the decoder using R-GCN message propagation."""
        # TODO: Start of commented out for debugging.
        # # use buffered messages if applicable
        # if self.enriched_embeddings is not None:
        #     return self.enriched_embeddings
        #
        # # clear cached embeddings as soon as possible to avoid unnecessary memory consumption
        # self.enriched_embeddings = None
        # TODO: End of commented out for debugging

        # Bind fields
        # shape: (num_entities, embedding_dim)
        x = self.entity_embeddings(indices=None)
        sources = self.sources
        targets = self.targets
        edge_types = self.edge_types

        # Edge dropout: drop the same edges on all layers (only in training mode)
        if self.training and self.edge_dropout is not None:
            # Get random dropout mask
            edge_keep_mask = torch.rand(self.sources.shape[0], device=x.device) > self.edge_dropout

            # Apply to edges
            sources = sources[edge_keep_mask]
            targets = targets[edge_keep_mask]
            edge_types = edge_types[edge_keep_mask]

        # Different dropout for self-loops (only in training mode)
        if self.training and self.self_loop_dropout is not None:
            node_keep_mask = torch.rand(self.num_entities, device=x.device) > self.self_loop_dropout
        else:
            node_keep_mask = None

        # fixed edges -> pre-compute weights
        if self.edge_weighting is not None:
            edge_weights = torch.empty_like(sources, dtype=torch.float32)
            for r in range(self.num_relations):
                mask = edge_types == r
                if mask.any():
                    edge_weights[mask] = self.edge_weighting(sources[mask], targets[mask])
        else:
            edge_weights = None

        for layer in self.layers:
            if isinstance(layer, RelationSpecificMessagePassing):
                kwargs = dict(
                    node_keep_mask=node_keep_mask,
                    source=sources,
                    target=targets,
                    edge_type=edge_types,
                    edge_weights=edge_weights,
                )
            else:
                kwargs = dict()
            x = layer(x, **kwargs)

        # TODO: Start of commented out for debugging.
        # # Cache enriched representations
        # self.enriched_embeddings = x
        # TODO: End of commented out for debugging

        return x


class Decoder(nn.Module):
    # TODO: Replace this by interaction function, once https://github.com/pykeen/pykeen/pull/107 is merged.
    def forward(self, h, r, t):
        return (h * r * t).sum(dim=-1)


class RGCN(Model):
    """An implementation of R-GCN from [schlichtkrull2018]_.

    This model uses graph convolutions with relation-specific weights.

    .. seealso::

       - `Pytorch Geometric's implementation of R-GCN
         <https://github.com/rusty1s/pytorch_geometric/blob/1.3.2/examples/rgcn.py>`_
       - `DGL's implementation of R-GCN
         <https://github.com/dmlc/dgl/tree/v0.4.0/examples/pytorch/rgcn>`_
    """

    #: The layers
    layers: Sequence[nn.Module]

    edge_weighting: Optional[EdgeWeighting]

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=1000, q=50),
        num_bases=dict(type=int, low=2, high=100, q=1),
        num_blocks=dict(type=int, low=2, high=20, q=1),
        num_layers=dict(type=int, low=1, high=5, q=1),
        use_bias=dict(type='bool'),
        use_batch_norm=dict(type='bool'),
        activation_cls=dict(type='categorical', choices=[None, nn.ReLU, nn.LeakyReLU]),
        base_model_cls=dict(type='categorical', choices=[DistMult, ComplEx, ERMLP]),
        edge_dropout=dict(type=float, low=0.0, high=.9),
        self_loop_dropout=dict(type=float, low=0.0, high=.9),
        edge_weighting=dict(type='categorical', choices=list(edge_weightings)),
        decomposition=dict(type='categorical', choices=list(decompositions)),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 500,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        num_layers: int = 2,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        activation_cls: Optional[Type[nn.Module]] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        sparse_messages_slcwa: bool = True,
        edge_dropout: float = 0.4,
        self_loop_dropout: float = 0.2,
        edge_weighting: Union[None, str, EdgeWeighting] = 'inverse_indegree',
        _decomposition: Union[None, str, Type[RelationSpecificMessagePassing]] = None,
        buffer_messages: bool = True,
        memory_intense: bool = False,
    ):
        """Initialize the model.

        :param triples_factory:
            The triples factory.
        :param embedding_dim:
            The embedding dimension to use. The same dimension is kept for all message passing layers.
        :param automatic_memory_optimization:
            Whether to apply automatic memory optimization for evaluation.
        :param loss:
            The loss function.
        :param predict_with_sigmoid:
            Whether to apply sigmoid on the model's output in evaluation mode.
        :param preferred_device:
            The preferred device.
        :param random_seed:
            The random seed used for initializing weights.
        :param num_bases: >0
            The number of bases. Requires decomposition=BasesDecomposition to become effective.
        :param num_blocks: >0
            The number of blocks. Requires decomposition=BlockDecomposition to become effective.
        :param num_layers:
            The number of layers.
        :param use_bias:
            Whether to use a bias.
        :param use_batch_norm:
            Whether to use batch normalization layers.
        :param activation_cls:
            The activation function to use.
        :param activation_kwargs:
            Additional key-word based parameters used to instantiate the activation layer.
        :param sparse_messages_owa:
            Whether to use sparse messages when training with OWA, i.e. do not compute representations for all nodes,
            but only those in a neighborhood of the currently considered ones. Theoretically improves memory
            requirements and runtime, since only a part of the messages are computed, but leads to additional masking.
            Moreover, real-world graphs often exhibit small-world properties leading to neighborhoods quickly comprising
            larger parts of the graph.
        :param edge_dropout:
            The edge dropout to use. Set to None to disable edge dropout.
        :param self_loop_dropout:
            The edge dropout to use for self-loops. Set to None to disable edge dropout.
        :param edge_weighting:
            The edge weighting function to use.
        :param decomposition:
            The decomposition of the relation-specific weight matrices.
        :param buffer_messages:
            Whether to buffer messages. Useful for instance in evaluation mode, when the parameters remain unchanged,
            but many forward passes are requested.
        :param memory_intense:
            Enable memory-intense forward pass which may be faster, in particular if the number of different relations
            is small.
        """
        if self.triples_factory.create_inverse_triples:
            raise ValueError('R-GCN handles edges in an undirected manner.')

        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        self.entity_representations = RGCNRepresentations(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            num_bases=num_bases,
            num_blocks=num_blocks,
            num_layers=num_layers,
            use_bias=use_bias,
            use_batch_norm=use_batch_norm,
            activation_cls=activation_cls,
            activation_kwargs=activation_kwargs,
            sparse_messages_slcwa=sparse_messages_slcwa,
            edge_dropout=edge_dropout,
            self_loop_dropout=self_loop_dropout,
            edge_weighting=edge_weighting,
            _decomposition=_decomposition,
            buffer_messages=buffer_messages,
            memory_intense=memory_intense,
        )
        self.relation_embeddings = Embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=embedding_dim,
        )
        # TODO: Dummy
        self.decoder = Decoder()

        # Finalize initialization, needs to be done manually instead of with
        # a post-init hook because this model is very special :)
        self.reset_parameters_()

    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()
        self.entity_representations.post_parameter_update()
        self.relation_embeddings.post_parameter_update()

    def _reset_parameters_(self):
        self.entity_representations.reset_parameters()
        self.relation_embeddings.reset_parameters()

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Enrich embeddings
        h = self.entity_representations(indices=hrt_batch[:, 0])
        t = self.entity_representations(indices=hrt_batch[:, 2])
        r = self.relation_embeddings(indices=hrt_batch[:, 1])
        return self.decoder(h, r, t).unsqueeze(dim=-1)
