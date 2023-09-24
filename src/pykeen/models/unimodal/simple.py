# -*- coding: utf-8 -*-

"""Implementation of SimplE."""

from typing import Any, ClassVar, Mapping, Optional, Tuple, Type, Union

from class_resolver import OptionalKwargs

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss, SoftplusLoss
from ...nn.modules import SimplEInteraction
from ...regularizers import PowerSumRegularizer, Regularizer, regularizer_resolver
from ...typing import Hint, Initializer

__all__ = [
    "SimplE",
]


class SimplE(ERModel):
    r"""An implementation of SimplE [kazemi2018]_.

    SimplE is an extension of canonical polyadic (CP), an early tensor factorization approach in which each entity
    $e \in \mathcal{E}$ is represented by two vectors $\textbf{h}_e, \textbf{t}_e \in \mathbb{R}^d$ and each
    relation by a single vector $\textbf{r}_r \in \mathbb{R}^d$. Depending whether an entity participates in a
    triple as the head or tail entity, either $\textbf{h}$ or $\textbf{t}$ is used. Both entity
    representations are learned independently, i.e. observing a triple $(h,r,t)$, the method only updates
    $\textbf{h}_h$ and $\textbf{t}_t$. In contrast to CP, SimplE introduces for each relation $\textbf{r}_r$
    the inverse relation $\textbf{r'}_r$, and formulates its the interaction model based on both:

    .. math::

        f(h,r,t) = \frac{1}{2}\left(\left\langle\textbf{h}_h, \textbf{r}_r, \textbf{t}_t\right\rangle
        + \left\langle\textbf{h}_t, \textbf{r'}_r, \textbf{t}_h\right\rangle\right)

    Therefore, for each triple $(h,r,t) \in \mathbb{K}$, both $\textbf{h}_h$ and $\textbf{h}_t$
    as well as $\textbf{t}_h$ and $\textbf{t}_t$ are updated.

    .. seealso::

       - Official implementation: https://github.com/Mehran-k/SimplE
       - Improved implementation in pytorch: https://github.com/baharefatemi/SimplE
    ---
    citation:
        author: Kazemi
        year: 2018
        link: https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs
        github: Mehran-k/SimplE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = SoftplusLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}
    #: The regularizer used by [trouillon2016]_ for SimplE
    #: In the paper, they use weight of 0.1, and do not normalize the
    #: regularization term by the number of elements, which is 200.
    regularizer_default: ClassVar[Type[Regularizer]] = PowerSumRegularizer
    #: The power sum settings used by [trouillon2016]_ for SimplE
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=20,
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        clamp_score: Optional[Union[float, Tuple[float, float]]] = None,
        entity_initializer: Hint[Initializer] = None,
        relation_initializer: Hint[Initializer] = None,
        regularizer: Hint[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            the embedding dimension
        :param clamp_score:
            whether to clamp scores, cf. :meth:`SimplEInteraction.__init__`
        :param entity_initializer:
            the entity representation initializer
        :param relation_initializer:
            the relation representation initializer
        :param regularizer:
            the regularizer, defaults to :attr:`SimplE.regularizer_default`
        :param regularizer_kwargs:
            additional keyword-based parameters passed to the regularizer, defaults to
            :attr:`SimplE.regularizer_default_kwargs`
        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        regularizer = regularizer_resolver.make_safe(regularizer, pos_kwargs=regularizer_kwargs)
        super().__init__(
            interaction=SimplEInteraction,
            interaction_kwargs=dict(clamp_score=clamp_score),
            entity_representations_kwargs=[
                # (head) entity
                dict(
                    shape=embedding_dim,
                    initializer=entity_initializer,
                    regularizer=regularizer,
                ),
                # tail entity
                dict(
                    shape=embedding_dim,
                    initializer=entity_initializer,
                    regularizer=regularizer,
                ),
            ],
            relation_representations_kwargs=[
                # relations
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    regularizer=regularizer,
                ),
                # inverse relations
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    regularizer=regularizer,
                ),
            ],
            **kwargs,
        )
