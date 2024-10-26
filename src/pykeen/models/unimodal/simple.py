"""Implementation of SimplE."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from class_resolver import OptionalKwargs

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss, SoftplusLoss
from ...nn.modules import Clamp, ClampedInteraction, SimplEInteraction
from ...regularizers import PowerSumRegularizer, Regularizer, regularizer_resolver
from ...typing import FloatTensor, Hint, Initializer

__all__ = [
    "SimplE",
]


class SimplE(
    ERModel[tuple[FloatTensor, FloatTensor], tuple[FloatTensor, FloatTensor], tuple[FloatTensor, FloatTensor]]
):
    r"""An implementation of SimplE [kazemi2018]_.

    SimplE learns two $d$-dimensional vectors for each entity and each relation, stored in
    :class:`~pykeen.nn.representation.Embedding`, and applies the :class:`~pykeen.nn.modules.SimplEInteraction` on top.

    .. note ::
        In the code in their repository, the score is clamped to $[-20, 20]$.
        That is not mentioned in the paper, so it is made optional.

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
    loss_default: ClassVar[type[Loss]] = SoftplusLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}
    #: The regularizer used by [trouillon2016]_ for SimplE
    #: In the paper, they use weight of 0.1, and do not normalize the
    #: regularization term by the number of elements, which is 200.
    regularizer_default: ClassVar[type[Regularizer]] = PowerSumRegularizer
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
        clamp_score: Clamp | float | None = None,
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
            whether to clamp scores, cf. :class:`~pykeen.nn.modules.ClampedInteraction`
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
        # TODO: what about using the default regularizer?
        regularizer = regularizer_resolver.make_safe(regularizer, pos_kwargs=regularizer_kwargs)
        # Note: In the code in their repository, the score is clamped to [-20, 20].
        #       That is not mentioned in the paper, so it is made optional here.
        super().__init__(
            interaction=ClampedInteraction,
            interaction_kwargs=dict(base=SimplEInteraction, clamp_score=clamp_score),
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
