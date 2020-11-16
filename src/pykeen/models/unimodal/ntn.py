# -*- coding: utf-8 -*-

"""Implementation of NTN."""

from typing import Optional

from torch import nn

from .. import Model
from ...losses import Loss
from ...nn import Embedding
from ...nn.modules import NTNInteraction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'NTN',
]


class NTN(Model):
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
        w = Embedding(
            num_embeddings=triples_factory.num_relations,
            shape=(num_slices, embedding_dim, embedding_dim),
        )
        b = Embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=num_slices,
        )
        u = Embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=num_slices,
        )
        vh = Embedding(
            num_embeddings=triples_factory.num_relations,
            shape=(num_slices, embedding_dim),
        )
        vt = Embedding(
            num_embeddings=triples_factory.num_relations,
            shape=(num_slices, embedding_dim),
        )
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            interaction=NTNInteraction(
                non_linearity=non_linearity,
            ),
            entity_representations=Embedding(
                num_embeddings=triples_factory.num_entities,
                embedding_dim=embedding_dim,
            ),
            relation_representations=(w, b, u, vh, vt),
        )
