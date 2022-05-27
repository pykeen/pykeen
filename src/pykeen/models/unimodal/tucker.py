# -*- coding: utf-8 -*-

"""Implementation of TuckEr."""

from typing import Any, ClassVar, Mapping, Optional, Type

from class_resolver import OptionalKwargs

from ..nbase import ERModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn import TuckerInteraction
from ...nn.init import xavier_normal_
from ...typing import Hint, Initializer

__all__ = [
    "TuckER",
]


class TuckER(ERModel):
    r"""An implementation of TuckEr from [balazevic2019]_.

    TuckER is a linear model that is based on the tensor factorization method Tucker in which a three-mode tensor
    $\mathfrak{X} \in \mathbb{R}^{I \times J \times K}$ is decomposed into a set of factor matrices
    $\textbf{A} \in \mathbb{R}^{I \times P}$, $\textbf{B} \in \mathbb{R}^{J \times Q}$, and
    $\textbf{C} \in \mathbb{R}^{K \times R}$ and a core tensor
    $\mathfrak{Z} \in \mathbb{R}^{P \times Q \times R}$ (of lower rank):

    .. math::

        \mathfrak{X} \approx \mathfrak{Z} \times_1 \textbf{A} \times_2 \textbf{B} \times_3 \textbf{C}

    where $\times_n$ is the tensor product, with $n$ denoting along which mode the tensor product is computed.
    In TuckER, a knowledge graph is considered as a binary tensor which is factorized using the Tucker factorization
    where $\textbf{E} = \textbf{A} = \textbf{C} \in \mathbb{R}^{n_{e} \times d_e}$ denotes the entity embedding
    matrix, $\textbf{R} = \textbf{B} \in \mathbb{R}^{n_{r} \times d_r}$ represents the relation embedding matrix,
    and $\mathfrak{W} = \mathfrak{Z} \in \mathbb{R}^{d_e \times d_r \times d_e}$ is the *core tensor* that
    indicates the extent of interaction between the different factors. The interaction model is defined as:

    .. math::

        f(h,r,t) = \mathfrak{W} \times_1 \textbf{h} \times_2 \textbf{r} \times_3 \textbf{t}

    where $\textbf{h},\textbf{t}$ correspond to rows of $\textbf{E}$ and $\textbf{r}$ to a row of $\textbf{R}$.

    The dropout values correspond to the following dropouts in the model's score function:

    .. math::

        \text{Dropout}_2(BN(\text{Dropout}_0(BN(h)) \times_1 \text{Dropout}_1(W \times_2 r))) \times_3 t

    where h,r,t are the head, relation, and tail embedding, W is the core tensor, \times_i denotes the tensor
    product along the i-th mode, BN denotes batch normalization, and :math:`\text{Dropout}` dropout.

    .. seealso::

       - Official implementation: https://github.com/ibalazevic/TuckER
       - pykg2vec implementation of TuckEr https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TuckER.py
    ---
    citation:
        author: Balažević
        year: 2019
        link: https://arxiv.org/abs/1901.09590
        github: ibalazevic/TuckER
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        relation_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        dropout_0=DEFAULT_DROPOUT_HPO_RANGE,
        dropout_1=DEFAULT_DROPOUT_HPO_RANGE,
        dropout_2=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        relation_dim: Optional[int] = None,
        dropout_0: float = 0.3,
        dropout_1: float = 0.4,
        dropout_2: float = 0.5,
        apply_batch_normalization: bool = True,
        entity_initializer: Hint[Initializer] = xavier_normal_,
        relation_initializer: Hint[Initializer] = xavier_normal_,
        core_tensor_initializer: Hint[Initializer] = None,
        core_tensor_initializer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            the (entity) embedding dimension
        :param relation_dim:
            the relation embedding dimension. Defaults to `embedding_dim`.
        :param dropout_0:
            the first dropout, cf. formula
        :param dropout_1:
            the second dropout, cf. formula
        :param dropout_2:
            the third dropout, cf. formula
        :param apply_batch_normalization:
            whether to apply batch normalization
        :param entity_initializer:
            the entity representation initializer
        :param relation_initializer:
            the relation representation initializer
        :param core_tensor_initializer:
            the core tensor initializer
        :param core_tensor_initializer_kwargs:
            keyword-based parameters passed to the core tensor initializer
        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        relation_dim = relation_dim or embedding_dim
        super().__init__(
            interaction=TuckerInteraction,
            interaction_kwargs=dict(
                embedding_dim=embedding_dim,
                relation_dim=relation_dim,
                head_dropout=dropout_0,  # TODO: rename
                relation_dropout=dropout_1,
                head_relation_dropout=dropout_2,
                apply_batch_normalization=apply_batch_normalization,
                core_initializer=core_tensor_initializer,
                core_initializer_kwargs=core_tensor_initializer_kwargs,
            ),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=dict(
                shape=relation_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
        )
