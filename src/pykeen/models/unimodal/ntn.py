# -*- coding: utf-8 -*-

"""Implementation of NTN."""

from typing import Any, ClassVar, Mapping, Optional

from class_resolver import Hint, HintOrType
from torch import nn

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.modules import NTNInteraction
from ...typing import Initializer

__all__ = [
    "NTN",
]


class NTN(ERModel):
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

    .. note::

        We split the original $V_r$ matrix into two parts, to separate $V_r [h; r] = V_r^h h + V_r^t t$.
        The latter is more efficient, if $h$ and $t$ are not of the same shape,
        e.g., since we are in a :meth:`score_h` / :meth:`score_t` setting.

    .. seealso::

       - Original Implementation (Matlab): `<https://github.com/khurram18/NeuralTensorNetworks>`_
       - TensorFlow: `<https://github.com/dddoss/tensorflow-socher-ntn>`_
       - Keras: `<https://github.com/dapurv5/keras-neural-tensor-layer (Keras)>`_
    ---
    citation:
        author: Socher
        year: 2013
        link: https://dl.acm.org/doi/10.5555/2999611.2999715
        github: khurram18/NeuralTensorNetworks
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        num_slices=dict(type=int, low=2, high=4),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 100,
        num_slices: int = 4,
        non_linearity: HintOrType[nn.Module] = None,
        non_linearity_kwargs: Optional[Mapping[str, Any]] = None,
        entity_initializer: Hint[Initializer] = None,
        **kwargs,
    ) -> None:
        r"""Initialize NTN.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 350]$.
        :param num_slices: The number of slices in the parameters
        :param non_linearity: A non-linear activation function. Defaults to the hyperbolic
            tangent :class:`torch.nn.Tanh`.
        :param non_linearity_kwargs: If the ``non_linearity`` is passed as a class, these keyword arguments
            are used during its instantiation.
        :param entity_initializer: Entity initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.EntityEmbeddingModel`
        """
        super().__init__(
            interaction=NTNInteraction(
                activation=non_linearity,
                activation_kwargs=non_linearity_kwargs,
            ),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=[
                # w: (k, d, d)
                dict(shape=(num_slices, embedding_dim, embedding_dim)),
                # vh: (k, d)
                dict(shape=(num_slices, embedding_dim)),
                # vt: (k, d)
                dict(shape=(num_slices, embedding_dim)),
                # b: (k,)
                dict(shape=(num_slices,)),
                # u: (k,)
                dict(shape=(num_slices,)),
            ],
            **kwargs,
        )
