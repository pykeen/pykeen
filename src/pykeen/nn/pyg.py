"""
Message Passing layers via PyTorch Geometric.

* RGCN
"""
from typing import Literal, Optional, Sequence, Union

import torch
from class_resolver import HintOrType, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver
from torch import nn

from .emb import EmbeddingSpecification, RepresentationModule
from ..triples import CoreTriplesFactory

try:
    from torch_geometric.nn import conv
except ImportError:
    conv = None

__all__ = [
    "RGCNRepresentations",
]

PyGAggregationType = Literal["mean", "max", "add"]


class RGCNRepresentations(RepresentationModule):
    """

    cf. https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn_link_pred.py
    """

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        embedding_specification: Optional[EmbeddingSpecification] = None,
        num_layers: int = 2,
        dims: Union[None, int, Sequence[int]] = None,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        memory_intense: bool = True,
        root_weight: bool = True,
        bias: bool = True,
        aggregation: PyGAggregationType = "mean",
        activation: HintOrType[nn.Module] = None,
        activation_kwargs: OptionalKwargs = None,
    ):
        if conv is None:
            raise ImportError
        if embedding_specification is None:
            embedding_specification = EmbeddingSpecification(embedding_dim=32)
        assert isinstance(embedding_specification.shape, tuple) and len(embedding_specification.shape) == 1
        entity_embeddings = embedding_specification.make(num_embeddings=triples_factory.num_entities)
        if dims is None:
            dims = embedding_specification.shape[-1]
        if isinstance(dims, int):
            dims = [dims] * num_layers
        elif len(dims) != num_layers:
            raise ValueError(num_layers, dims)

        super().__init__(max_id=entity_embeddings.max_id, shape=(dims[-1],))

        # has to be assigned *after* super.__init__ has been called
        self.entity_embeddings = entity_embeddings
        self.activation = activation_resolver.make(activation, activation_kwargs)

        # create message passing layers
        layers = []
        layer_cls = conv.FastRGCNConv if memory_intense else conv.RGCNConv
        for input_dim, output_dim in zip(dims, dims[1:]):
            layers.append(
                layer_cls(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    num_relations=triples_factory.num_relations,
                    num_bases=num_bases,
                    num_blocks=num_blocks,
                    aggr=aggregation,
                    root_weight=root_weight,
                    bias=bias,
                )
            )
        self.layers = nn.ModuleList(layers)

        # register buffers
        self.register_buffer(name="edge_index", tensor=triples_factory.edge_index)
        self.register_buffer(name="edge_type", tensor=triples_factory.edge_type)

        # buffering
        self.enriched_embeddings = None

    def _real_forward(self) -> torch.FloatTensor:
        if self.enriched_embeddings is not None:
            return self.enriched_embeddings

        x = self.entity_embeddings(indices=None)
        for i, layer in enumerate(self.layers):
            x = layer(
                x=x,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
            )
            # no activation on last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)

        # Cache enriched representations
        self.enriched_embeddings = x

        return x

    def forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Enrich the entity embeddings of the decoder using R-GCN message propagation."""
        x = self._real_forward()
        if indices is not None:
            x = x[indices]
        return x
