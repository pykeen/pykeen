"""
PyTorch Geometric based representation modules.

The modules enable entity representations which are linked to their graph neighbors' representations. Similar
representations are those by CompGCN or R-GCN. However, this module offers generic modules to combine many of the
numerous message passing layers from PyTorch Geometric with base representations. A summary of available message passing
layers can be found at :mod:`torch_geometric.nn.conv`.

The three classes differ in how the make use of the relation type information:

* :class:`SimpleMessagePassingRepresentation` only uses the connectivity information from the training triples,
  but ignores the relation type, e.g., :class:`torch_geometric.nn.conv.GCNConv`.
* :class:`TypedMessagePassingRepresentation` is for message passing layer, which internally handle the
  categorical relation type information via an `edge_type` input, e.g., :class:`torch_geometric.nn.conv.RGCNConv`.
* :class:`FeaturizedMessagePassingRepresentation` is for message passing layer which can use edge attributes
  via the parameter `edge_attr`, e.g., :class:`torch_geometric.nn.conv.GMMConv`.

We can also easily utilize these representations with :class:`pykeen.models.ERModel`. Here, we showcase how to combine
static label-based entity features with a trainable GCN encoder for entity representations, with learned embeddings for
relation representations and a DistMult interaction function.

.. code-block:: python

    from pykeen.datasets import get_dataset
    from pykeen.models import ERModel
    from pykeen.nn.init import LabelBasedInitializer
    from pykeen.pipeline import pipeline

    dataset = get_dataset(dataset="nations", dataset_kwargs=dict(create_inverse_triples=True))
    entity_initializer = LabelBasedInitializer.from_triples_factory(
        triples_factory=dataset.training,
        for_entities=True,
    )
    (embedding_dim,) = entity_initializer.tensor.shape[1:]
    r = pipeline(
        dataset=dataset,
        model=ERModel,
        model_kwargs=dict(
            interaction="distmult",
            entity_representations="SimpleMessagePassing",
            entity_representations_kwargs=dict(
                triples_factory=dataset.training,
                base_kwargs=dict(
                    shape=embedding_dim,
                    initializer=entity_initializer,
                    trainable=False,
                ),
                layers=["GCN"] * 2,
                layers_kwargs=dict(in_channels=embedding_dim, out_channels=embedding_dim),
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
            ),
        ),
    )
"""

from abc import ABC, abstractmethod
from typing import Collection, Literal, Optional, Sequence

import torch
from class_resolver import ClassResolver, HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver
from torch import nn

from .representation import Representation
from .utils import ShapeError
from ..triples.triples_factory import CoreTriplesFactory
from ..typing import OneOrSequence
from ..utils import get_edge_index, upgrade_to_sequence

__all__ = [
    # abstract
    "MessagePassingRepresentation",
    # concrete classes
    "SimpleMessagePassingRepresentation",
    "FeaturizedMessagePassingRepresentation",
    "TypedMessagePassingRepresentation",
]

try:
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.utils import k_hop_subgraph

    layer_resolver: ClassResolver[MessagePassing] = ClassResolver.from_subclasses(
        base=MessagePassing,  # type: ignore
        suffix="Conv",
    )
except ImportError:
    MessagePassing = None
    layer_resolver = None
    k_hop_subgraph = None

_PYG_INSTALLATION_TEXT = """
Requires `torch_geometric` to be installed.

Please refer to

    https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

for installation instructions.
"""

FlowDirection = Literal["source_to_target", "target_to_source"]
FLOW_DIRECTIONS: Collection[FlowDirection] = {"source_to_target", "target_to_source"}


def _extract_flow(layers: Sequence[MessagePassing]) -> FlowDirection:
    """Extract the flow direction from the message passing layers."""
    flow: Optional[FlowDirection] = None
    for layer in layers:
        if flow is None:
            if layer.flow not in FLOW_DIRECTIONS:
                raise AssertionError(f"Invalid flow: {layer.flow}. Valid flows: {FLOW_DIRECTIONS}")
            flow = layer.flow
        elif flow != layer.flow:
            raise ValueError(f"Different flow directions across layers: {[l.flow for l in layers]}")
    # default flow
    return flow or "source_to_target"


class MessagePassingRepresentation(Representation, ABC):
    """
    An abstract representation class utilizing PyTorch Geometric message passing layers.

    It comprises:
        * base (entity) representations, which can also be passed as hints
        * a sequence of message passing layers. They are utilized in an abstract
          :meth:`MessagePassingRepresentation._message_passing` to enrich the base representations
          by neighborhood information.
        * a sequence of activation layers in between the message passing layers.
        * an `edge_index` buffer, which stores the edge index and is moved to the device alongside
          the module.
    """

    #: the message passing layers
    layers: Sequence[MessagePassing]

    #: the flow direction of messages across layers
    flow: FlowDirection

    #: the edge index, shape: (2, num_edges)
    edge_index: torch.LongTensor

    def __init__(
        self,
        triples_factory: CoreTriplesFactory,
        layers: OneOrManyHintOrType[MessagePassing],
        layers_kwargs: OneOrManyOptionalKwargs = None,
        base: HintOrType[Representation] = None,
        base_kwargs: OptionalKwargs = None,
        max_id: Optional[int] = None,
        shape: Optional[OneOrSequence[int]] = None,
        activations: OneOrManyHintOrType[nn.Module] = None,
        activations_kwargs: OneOrManyOptionalKwargs = None,
        restrict_k_hop: bool = False,
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

        :param shape:
            the output shape. Defaults to the base representation shape. Has to match to output shape of the last
            message passing layer.
        :param max_id:
            the number of representations. If provided, has to match the base representation's max_id

        :param activations:
            the activation(s), or hints thereof
        :param activations_kwargs:
            additional keyword-based parameters passed to the activations upon instantiation
        :param restrict_k_hop:
            whether to restrict the message passing only to the k-hop neighborhood, when only some indices
            are requested. This utilizes :func:`torch_geometric.utils.k_hop_subgraph`.

        :param kwargs:
            additional keyword-based parameters passed to :meth:`Representation.__init__`

        :raises ImportError:
            if PyTorch Geometric is not installed
        :raises ValueError:
            if the number of activations and message passing layers do not match (after input normalization)
        """
        # fail if dependencies are missing
        if MessagePassing is None or layer_resolver is None or k_hop_subgraph is None:
            raise ImportError(_PYG_INSTALLATION_TEXT)

        # avoid cyclic import
        from . import representation_resolver

        # the base representations, e.g., entity embeddings or features
        base = representation_resolver.make(base, pos_kwargs=base_kwargs, max_id=triples_factory.num_entities)

        # verify max_id
        max_id = max_id or base.max_id
        if max_id != base.max_id:
            raise ValueError(f"Inconsistent max_id={max_id} vs. base.max_id={base.max_id}")

        # verify shape
        shape = ShapeError.verify(shape=shape or base.shape, reference=shape)
        super().__init__(max_id=max_id, shape=shape, **kwargs)

        # assign sub-module *after* super call
        self.base = base

        # initialize layers
        self.layers = nn.ModuleList(layer_resolver.make_many(layers, layers_kwargs))
        self.flow = _extract_flow(self.layers)

        # normalize activation
        activations = list(upgrade_to_sequence(activations))
        if len(activations) == 1:
            activations = activations * len(self.layers)
        self.activations = nn.ModuleList(activation_resolver.make_many(activations, activations_kwargs))

        # check consistency
        if len(self.layers) != len(self.activations):
            raise ValueError(
                f"The lengths of the list of message passing layers ({len(self.layers)}) "
                f"and activation layers ({len(self.activation)}) differs! To disable activations "
                f"on certain layers, e.g., the last, use torch.nn.Identity."
            )

        # buffer edge index for message passing
        self.register_buffer(name="edge_index", tensor=get_edge_index(triples_factory=triples_factory))

        self.restrict_k_hop = restrict_k_hop

        # TODO: inductiveness; we need to
        #   * replace edge_index
        #   * replace base representations
        #   * keep layers & activations

    # docstr-coverage: inherited
    def _plain_forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa: D102
        if self.restrict_k_hop and indices is not None:
            # we can restrict the message passing to the k-hop neighborhood of the desired indices;
            # this does only make sense if we do not request *all* indices
            assert k_hop_subgraph is not None
            # k_hop_subgraph returns:
            # (1) the nodes involved in the subgraph
            # (2) the filtered edge_index connectivity
            # (3) the mapping from node indices in node_idx to their new location, and
            # (4) the edge mask indicating which edges were preserved
            neighbor_indices, edge_index, indices, edge_mask = k_hop_subgraph(
                node_idx=indices,
                num_hops=len(self.layers),
                edge_index=self.edge_index,
                relabel_nodes=True,
                flow=self.flow,
            )
            # we only need the base representations for the neighbor indices
            x = self.base(indices=neighbor_indices)
        else:
            # get *all* base representations
            x = self.base(indices=None)
            # use *all* edges
            edge_index = self.edge_index
            edge_mask = None
        # perform message passing
        x = self.pass_messages(x=x, edge_index=edge_index, edge_mask=edge_mask)
        # select desired indices
        if indices is not None:
            x = x[indices]
        return x

    @abstractmethod
    def pass_messages(
        self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        """
        Perform the message passing steps.

        :param x: shape: `(n, d_in)`
            the base entity representations
        :param edge_index: shape: `(num_selected_edges,)`
            the edge index (which may already be a selection of the full edge index)
        :param edge_mask: shape: `(num_edges,)`
            an edge mask if message passing is restricted

        :return: shape: `(n, d_out)`
            the enriched entity representations
        """
        raise NotImplementedError


class SimpleMessagePassingRepresentation(MessagePassingRepresentation):
    """
    A representation with message passing not making use of the relation type.

    By only using the connectivity information, but not the relation type information, this module
    can utilize message passing layers defined on uni-relational graphs, which are the majority of
    available layers from the PyTorch Geometric library.

    Here, we create a two-layer :class:`torch_geometric.nn.conv.GCNConv` on top of an
    :class:`pykeen.nn.representation.Embedding`:

    .. code-block:: python

        from pykeen.datasets import get_dataset

        embedding_dim = 64
        dataset = get_dataset(dataset="nations")
        r = SimpleMessagePassingRepresentation(
            triples_factory=dataset.training,
            base_kwargs=dict(shape=embedding_dim),
            layers=["gcn"] * 2,
            layers_kwargs=dict(in_channels=embedding_dim, out_channels=embedding_dim),
        )
    """

    # docstr-coverage: inherited
    def pass_messages(
        self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:  # noqa: D102
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x, edge_index=edge_index))
        return x


class TypedMessagePassingRepresentation(MessagePassingRepresentation):
    """
    A representation with message passing with uses categorical relation type information.

    The message passing layers of this module internally handle the categorical relation type information
    via an `edge_type` input, e.g., :class:`torch_geometric.nn.conv.RGCNConv`, or
    :class:`torch_geometric.nn.conv.RGATConv`.

    The following example creates a one-layer RGCN using the basis decomposition:

    .. code-block:: python

        from pykeen.datasets import get_dataset

        embedding_dim = 64
        dataset = get_dataset(dataset="nations")
        r = TypedMessagePassingRepresentation(
            triples_factory=dataset.training,
            base_kwargs=dict(shape=embedding_dim),
            layers="rgcn",
            layers_kwargs=dict(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                num_bases=2,
                num_relations=dataset.num_relations,
            ),
        )
    """

    #: the edge type, shape: (num_edges,)
    edge_type: torch.LongTensor

    def __init__(self, triples_factory: CoreTriplesFactory, **kwargs):
        """
        Initialize the representation.

        :param triples_factory:
            the factory comprising the training triples used for message passing
        :param kwargs:
            additional keyword-based parameters passed to :meth:`MessagePassingRepresentation.__init__`
        """
        super().__init__(triples_factory=triples_factory, **kwargs)
        # register an additional buffer for the categorical edge type
        self.register_buffer(name="edge_type", tensor=triples_factory.mapped_triples[:, 1])

    def _get_edge_type(self, edge_mask: Optional[torch.BoolTensor] = None) -> torch.LongTensor:
        """
        Return the (selected part of the) edge type.

        :param edge_mask: shape: `(num_edges,)`
            the edge mask

        :return: shape: `(num_selected_edges,)`
            the selected edge types, or all edge types if `edge_mask` is None
        """
        if edge_mask is None:
            return self.edge_type
        return self.edge_type[edge_mask]

    # docstr-coverage: inherited
    def pass_messages(
        self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:  # noqa: D102
        edge_type = self._get_edge_type(edge_mask=edge_mask)
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x, edge_index=edge_index, edge_type=edge_type))
        return x


class FeaturizedMessagePassingRepresentation(TypedMessagePassingRepresentation):
    """
    A representation with message passing with uses edge features obtained from relation representations.

    It (re-)uses a representation layer for relations to obtain edge features, which are then utilized
    by appropriate message passing layers, e.g., :class:`torch_geometric.nn.conv.GMMConv`, or
    :class:`torch_geometric.nn.conv.GATConv`. We further allow a (shared) transformation of edge features
    between layers.

    The following example creates a two-layer GAT on top of the base representations:


    .. code-block:: python

        from pykeen.datasets import get_dataset

        embedding_dim = 64
        dataset = get_dataset(dataset="nations")
        r = FeaturizedMessagePassingRepresentation(
            triples_factory=dataset.training,
            base_kwargs=dict(shape=embedding_dim),
            relation_representation_kwargs=dict(
                shape=embedding_dim,
            ),
            layers="gat",
            layers_kwargs=dict(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                edge_dim=embedding_dim,  # should match relation dim
            ),
        )
    """

    #: the relation representations used to obtain initial edge features
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
            additional keyword-based parameters passed to
            :meth:`TypedMessagePassingRepresentation.__init__`, except the `triples_factory`
        """
        super().__init__(triples_factory=triples_factory, **kwargs)

        # avoid cyclic import
        from . import representation_resolver

        self.relation_representation = representation_resolver.make(
            relation_representation, pos_kwargs=relation_representation_kwargs, max_id=triples_factory.num_relations
        )
        self.relation_transformation = relation_transformation

    # docstr-coverage: inherited
    def pass_messages(
        self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:  # noqa: D102
        edge_type = self._get_edge_type(edge_mask=edge_mask)
        # get initial relation representations
        x_rel = self.relation_representation(indices=None)
        n_layer = len(self.layers)
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            # select edge attributes from relation representations according to relation type
            edge_attr = x_rel[edge_type]
            # perform message passing
            x = activation(layer(x, edge_index=edge_index, edge_attr=edge_attr))
            # apply relation transformation, if necessary
            if self.relation_transformation is not None and i < n_layer - 1:
                x_rel = self.relation_transformation(x_rel)
        return x
