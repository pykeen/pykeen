"""
PyTorch Geometric based representation modules.

The modules enable entity representations which are linked to their graph neighbors' representations. Similar
representations are those by CompGCN or R-GCN. However, this module offers generic modules to combine many of the
numerous message passing layers from PyTorch Geometric with base representations. A summary of available message passing
layers can be found at https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html.

The three classes differ in how the make use of the relation type information:

* :class:`UniRelationalMessagePassingRepresentation` only uses the connectivity information from the training triples,
  but ignores the relation type, e.g., :class:`torch_geometric.nn.conv.GCNConv`.
* :class:`CategoricalRelationTypeMessagePassingRepresentation` is for message passing layer, which internally handle the
  categorical relation type information via an `edge_type` input, e.g., :class:`torch_geometric.nn.conv.RGCNConv`.
* :class:`FeaturizedRelationTypeMessagePassingRepresentation` is for message passing layer which can use edge attributes
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
            entity_representations="UniRelationalMessagePassing",
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
from abc import abstractmethod
from typing import Optional, Sequence

import torch
from class_resolver import ClassResolver, HintOrType, OneOrManyHintOrType, OneOrManyOptionalKwargs, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver
from torch import nn

from .representation import Representation
from ..triples.triples_factory import CoreTriplesFactory
from ..typing import OneOrSequence
from ..utils import upgrade_to_sequence

__all__ = [
    "UniRelationalMessagePassingRepresentation",
    "FeaturizedRelationTypeMessagePassingRepresentation",
    "CategoricalRelationTypeMessagePassingRepresentation",
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


class MessagePassingRepresentation(Representation):
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

        super().__init__(max_id=kwargs.pop("max_id", base.max_id), shape=output_shape or base.shape, **kwargs)

        # assign sub-module *after* super call
        self.base = base

        # initialize layers
        self.layers = nn.ModuleList(layer_resolver.make_many(layers, layers_kwargs))

        # normalize activation
        activation = upgrade_to_sequence(activation)
        if len(activation) == 1:
            activation = activation * len(self.layers)
        self.activations = nn.ModuleList(activation_resolver.make_many(activation, activation_kwargs))
        if len(self.layers) != len(self.activations):
            raise ValueError(
                f"The lengths of the list of message passing layers ({len(self.layers)}) "
                f"and activation layers ({len(self.activation)}) differs! To disable activations "
                f"on certain layers, e.g., the last, use torch.nn.Identity."
            )

        # prepare buffer
        # TODO: inductive?
        self.register_buffer(name="edge_index", tensor=triples_factory.mapped_triples[:, [0, 2]].t())

    def _plain_forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:  # noqa: D102
        # TODO: we could reduce the memory footprint and maybe also computation time
        #       by considering only the k-hop neighborhood of the requested indices
        #       computing this may be computationally expensive, too
        # get *all* base representations
        x = self.base(indices=None)
        # perform message passing on *all* base representations & edges
        x = self._message_passing(x=x)
        # select desired indices
        if indices is not None:
            x = x[indices]
        return x

    @abstractmethod
    def _message_passing(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Perform the message passing steps.

        :param x: shape: `(num_entities, *input_dims)`
            the base entity representations

        :return: shape: `(num_entities, *output_dims)`
            the enriched entity representations
        """
        raise NotImplementedError


class UniRelationalMessagePassingRepresentation(MessagePassingRepresentation):
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
        r = UniRelationalMessagePassingRepresentation(
            triples_factory=dataset.training,
            base_kwargs=dict(shape=embedding_dim),
            layers=["gcn"] * 2,
            layers_kwargs=dict(in_channels=embedding_dim, out_channels=embedding_dim),
        )
    """

    def _message_passing(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x, edge_index=self.edge_index))
        return x


class CategoricalRelationTypeMessagePassingRepresentation(MessagePassingRepresentation):
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
        r = CategoricalRelationTypeMessagePassingRepresentation(
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

    def _message_passing(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x, edge_index=self.edge_index, edge_type=self.edge_type))
        return x


class FeaturizedRelationTypeMessagePassingRepresentation(CategoricalRelationTypeMessagePassingRepresentation):
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
        r = FeaturizedRelationTypeMessagePassingRepresentation(
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
            :meth:`CategoricalRelationTypeMessagePassingRepresentation.__init__`, except the `triples_factory`
        """
        super().__init__(triples_factory=triples_factory, **kwargs)

        # avoid cyclic import
        from . import representation_resolver

        self.relation_representation = representation_resolver.make(
            relation_representation, pos_kwargs=relation_representation_kwargs, max_id=triples_factory.num_relations
        )
        self.relation_transformation = relation_transformation

    def _message_passing(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        # get initial relation representations
        x_rel = self.relation_representation(indices=None)
        n_layer = len(self.layers)
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            # select edge attributes from relation representations according to relation type
            edge_attr = x_rel[self.edge_type]
            # perform message passing
            x = activation(layer(x, edge_index=self.edge_index, edge_attr=edge_attr))
            # apply relation transformation, if necessary
            if self.relation_transformation is not None and i < n_layer - 1:
                x_rel = self.relation_transformation(x_rel)
        return x
