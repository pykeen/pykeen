"""PyTorch Geometric based representation modules."""
from abc import abstractmethod
from typing import Optional, Sequence

import torch
from class_resolver import ClassResolver, HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver
from torch import nn

from .representation import Representation
from ..triples.triples_factory import CoreTriplesFactory
from ..typing import OneOrSequence

__all__ = [
    "IgnoreRelationTypePyGRepresentation",
    "FeaturizedRelationTypePyGRepresentation",
    "CategoricalRelationTypePyGRepresentation",
]

try:
    from torch_geometric.nn.conv import MessagePassing

    layer_resolver = ClassResolver.from_subclasses(
        base=MessagePassing,  # type: ignore
        suffix="Conv",
    )
except ImportError:
    MessagePassing = None
    layer_resolver = None

_PYG_INSTALLATION_TEXT = """
Requires `torch_geometric` to be installed.

Please refer to 

    https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

for installation instructions.  
"""


class AbstractPyGRepresentation(Representation):
    """An abstract representation class utilizing PyTorch Geometric message passing layers."""

    #: the message passing layers
    layers: Sequence[MessagePassing]

    #: the edge index, shape: (2, num_edges)
    edge_index: torch.LongTensor

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        layers: OneOrManyHintOrType[MessagePassing],
        layers_kwargs: OneOrManyOptionalKwargs = None,
        base: HintOrType[Representation] = None,
        base_kwargs: OptionalKwargs = None,
        output_shape: OneOrSequence[int] = None,
        activation: OneOrManyHintOrType[nn.Module] = None,
        activation_kwargs: OneOrManyOptionalKwargs = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        :param triples_factory:
            the factory comprising the training triples used for message passing

        :param layers:
            the message passing layer(s) or hints thereof
        :param layers_kwargs:
            additional keyword-based parameters passed to the layers upon instantiation

        :param base:
            the base representations for entities, or a hint thereof
        :param base_kwargs:
            additional keyword-based parameters passed to the base representations upon instantiation

        :param output_shape:
            the output shape. Defaults to the base representation shape. Has to match to output shape of the last
            message passing layer.

        :param activation:
            the activation(s), or hints thereof
        :param activation_kwargs:
            additional keyword-based parameters passed to the activations upon instantiation

        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`

        :raises ImportError:
            if PyTorch Geometric is not installed
        """
        # fail if dependencies are missing
        if MessagePassing is None or layer_resolver is None:
            raise ImportError(_PYG_INSTALLATION_TEXT)

        # avoid cyclic import
        from . import representation_resolver

        # the base representations, e.g., entity embeddings or features
        base = representation_resolver.make(base, pos_kwargs=base_kwargs, max_id=triples_factory.num_entities)

        super().__init__(max_id=base.max_id, shape=output_shape or base.shape, **kwargs)

        # assign sub-module *after* super call
        self.base = base

        # initialize layers
        self.layers = nn.ModuleList(layer_resolver.make_many(layers, layers_kwargs))
        if activation is None:
            activation = [None] * len(self.layers)
        self.activations = nn.ModuleList(activation_resolver.make_many(activation, activation_kwargs))
        assert len(self.layers) == len(self.activations)

        # prepare buffer
        # TODO: inductive?
        self.register_buffer(name="edge_index", tensor=triples_factory.mapped_triples[:, [0, 2]].t())

    def _plain_forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa: D102
        x = self.base(indices=None)
        x = self._message_passing(x=x)
        if indices is not None:
            x = x[indices]
        return x

    @abstractmethod
    def _message_passing(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Perform the message passing."""
        raise NotImplementedError


class IgnoreRelationTypePyGRepresentation(AbstractPyGRepresentation):
    """A representation with message passing not making use of the relation type."""

    def _message_passing(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x, edge_index=self.edge_index))
        return x


class CategoricalRelationTypePyGRepresentation(AbstractPyGRepresentation):
    """A representation with message passing with uses categorical relation type information, e.g., R-GCN."""

    #: the edge type
    edge_type: torch.LongTensor

    def __init__(self, triples_factory: CoreTriplesFactory, **kwargs):
        """
        Initialize the representation.

        :param triples_factory:
            the factory comprising the training triples used for message passing
        :param kwargs:
            additional keyword-based parameters passed to :meth:`AbstractPyGRepresentation.__init__`
        """
        super().__init__(triples_factory=triples_factory, **kwargs)
        self.register_buffer(name="edge_type", tensor=triples_factory.mapped_triples[:, 1])

    def _message_passing(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x, edge_index=self.edge_index, edge_type=self.edge_type))
        return x


class FeaturizedRelationTypePyGRepresentation(CategoricalRelationTypePyGRepresentation):
    """A representation with message passing with uses categorical relation type information, e.g., R-GCN."""

    #: the edge type
    relation_representation: Representation

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        relation_representation: HintOrType[Representation] = None,
        relation_representation_kwargs: OptionalKwargs = None,
        relation_transformation: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        Initialize the representation.

        :param triples_factory:
            the factory comprising the training triples used for message passing

        :param relation_representation:
            the base representations for relations, or a hint thereof
        :param relation_representation_kwargs:
            additional keyword-based parameters passed to the base representations upon instantiation

        :param relation_transformation:
            an optional transformation to apply to the relation representations after each message passing step.
            If None, do not modify the representations.

        :param kwargs:
            additional keyword-based parameters passed to :meth:`AbstractPyGRepresentation.__init__`
        """
        super().__init__(triples_factory=triples_factory, **kwargs)
        # avoid cyclic import
        from . import representation_resolver

        self.relation_representation = representation_resolver.make(
            relation_representation, pos_kwargs=relation_representation_kwargs, max_id=triples_factory.num_relations
        )
        self.relation_transformation = relation_transformation

    def _message_passing(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        edge_attr = self.relation_representation(self.edge_type)
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x, edge_index=self.edge_index, edge_attr=edge_attr))
            if self.relation_transformation is not None:
                edge_attr = self.relation_transformation(edge_attr)
        return x
