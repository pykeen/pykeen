# -*- coding: utf-8 -*-

"""Implementation of NTN."""

from typing import Optional

import torch
from torch import nn

from ..base import EntityEmbeddingModel
from ...losses import Loss
from ...nn import Embedding
from ...nn.modules import NTNInteractionFunction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'NTN',
]


class NTN(EntityEmbeddingModel):
    r"""An implementation of NTN from [socher2013]_.

    NTN uses a bilinear tensor layer instead of a standard linear neural network layer:

    .. math::

        f(h,r,t) = \textbf{u}_{r}^{T} \cdot \tanh(\textbf{h} \mathfrak{W}_{r} \textbf{t}
        + \textbf{V}_r [\textbf{h};\textbf{t}] + \textbf{b}_r)

    where $\mathfrak{W}_r \in \mathbb{R}^{d \times d \times k}$ is the relation specific tensor, and the weight
    matrix $\textbf{V}_r \in \mathbb{R}^{k \times 2d}$, and the bias vector $\textbf{b}_r$ and
    the weight vector $\textbf{u}_r \in \mathbb{R}^k$ are the standard
    parameters of a neural network, which are also relation specific. The result of the tensor product
    $\textbf{h} \mathfrak{W}_{r} \textbf{t}$ is a vector $\textbf{x} \in \mathbb{R}^k$ where each entry $x_i$ is
    computed based on the slice $i$ of the tensor $\mathfrak{W}_{r}$:
    $\textbf{x}_i = \textbf{h}\mathfrak{W}_{r}^{i} \textbf{t}$. As indicated by the interaction model, NTN defines
    for each relation a separate neural network which makes the model very expressive, but at the same time
    computationally expensive.

    .. seealso::

       - Original Implementation (Matlab): `<https://github.com/khurram18/NeuralTensorNetworks>`_
       - TensorFlow: `<https://github.com/dddoss/tensorflow-socher-ntn>`_
       - Keras: `<https://github.com/dapurv5/keras-neural-tensor-layer (Keras)>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
        num_slices=dict(type=int, low=2, high=4),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 100,
        automatic_memory_optimization: Optional[bool] = None,
        num_slices: int = 4,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        non_linearity: Optional[nn.Module] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize NTN.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 350]$.
        :param num_slices:
        :param non_linearity: A non-linear activation function. Defaults to the hyperbolic
         tangent :class:`torch.nn.Tanh`.
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.num_slices = num_slices

        self.w = Embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=num_slices * self.embedding_dim ** 2,
        )
        self.b = Embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=num_slices,
        )
        self.u = Embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=num_slices,
        )
        self.vh = Embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=num_slices * embedding_dim,
        )
        self.vt = Embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=num_slices * embedding_dim,
        )
        self.interaction = NTNInteractionFunction(non_linearity=non_linearity)

    def forward(
        self,
        h_indices: Optional[torch.LongTensor] = None,
        r_indices: Optional[torch.LongTensor] = None,
        t_indices: Optional[torch.LongTensor] = None,
        slice_size: int = None,
    ) -> torch.FloatTensor:
        """
        Compute scores for NTN.

        :param h_indices: shape: (batch_size,)
        :param r_indices: shape: (batch_size,)
        :param t_indices: shape: (batch_size,)

        :return: shape: (batch_size, num_heads, num_relations, num_tails)
        """
        assert slice_size is None, "not implemented"

        #: shape: (batch_size, num_entities, d)
        h = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        t = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)
        w = self.w.get_in_canonical_shape(indices=r_indices,
                                          reshape_dim=(self.num_slices, self.embedding_dim, self.embedding_dim))
        b = self.b.get_in_canonical_shape(indices=r_indices)
        u = self.u.get_in_canonical_shape(indices=r_indices)
        vh = self.vh.get_in_canonical_shape(indices=r_indices, reshape_dim=(self.num_slices, self.embedding_dim))
        vt = self.vt.get_in_canonical_shape(indices=r_indices, reshape_dim=(self.num_slices, self.embedding_dim))

        return self.interaction(h=h, t=t, r=(w, b, u, vh, vt))
