"""Implementation of NTN."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

from class_resolver import Hint, HintOrType, ResolverKey, update_docstring_with_resolver_keys
from torch import nn

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.modules import NTNInteraction
from ...typing import FloatTensor, Initializer

__all__ = [
    "NTN",
]


class NTN(ERModel[FloatTensor, tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor], FloatTensor]):
    r"""An implementation of NTN from [socher2013]_.

    NTN represents entities using a $d$-dimensional vector.
    Relations are represented by

        - a $k \times d \times d$-dimensional tensor, $\mathbf{W} \in \mathbb{R}^{k \times d \times d}$,
        - a $2k \times d$-dimensional matrix, $\mathbf{V} \in \mathbb{R}^{k \times 2d}$, and
        - two $k$-dimensional vectors, $\mathbf{b}, \mathbf{u} \in \mathbb{R}^{k}$.

    Denoting the number of entities by $E$ and the number of relations by $R$, the total number of parameters is thus
    given by

    .. math ::

        dE + k(d^2 + 2d + 2)R

    All representations are stored as :class:`~pykeen.nn.representation.Embedding`.
    :class:`~pykeen.nn.modules.NTNInteraction` is used as interaction upon those representations.

    .. note::

        We split the original $k \times 2d$-dimensional $\mathbf{V}$ matrix into two parts of shape $k \times d$ to
        support more efficient 1:n scoring, e.g., in the :meth:`~pykeen.models.Model.score_h` or
        :meth:`~pykeen.models.Model.score_t` setting.

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

    @update_docstring_with_resolver_keys(
        ResolverKey(name="non_linearity", resolver="class_resolver.contrib.torch.activation_resolver")
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
            Remaining keyword arguments to forward to :class:`~pykeen.models.ERModel`
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
