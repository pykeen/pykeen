# -*- coding: utf-8 -*-

"""An implementation of TransH."""

import itertools
from typing import Any, ClassVar, Mapping, Type

from class_resolver import HintOrType, OptionalKwargs
from torch.nn import functional, init

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import TransHInteraction
from ...regularizers import NormLimitRegularizer, OrthogonalityRegularizer, Regularizer
from ...typing import Hint, Initializer

__all__ = [
    "TransH",
]


class TransH(ERModel):
    r"""An implementation of TransH [wang2014]_.

    This model extends :class:`pykeen.models.TransE` by applying the translation from head to tail entity in a
    relational-specific hyperplane in order to address its inability to model one-to-many, many-to-one, and
    many-to-many relations.

    In TransH, each relation is represented by a hyperplane, or more specifically a normal vector of this hyperplane
    $\textbf{w}_{r} \in \mathbb{R}^d$ and a vector $\textbf{d}_{r} \in \mathbb{R}^d$ that lies in the hyperplane.
    To compute the plausibility of a triple $(h,r,t)\in \mathbb{K}$, the head embedding $\textbf{e}_h \in \mathbb{R}^d$
    and the tail embedding $\textbf{e}_t \in \mathbb{R}^d$ are first projected onto the relation-specific hyperplane:

    .. math::

        \textbf{e'}_{h,r} = \textbf{e}_h - \textbf{w}_{r}^\top \textbf{e}_h \textbf{w}_r

        \textbf{e'}_{t,r} = \textbf{e}_t - \textbf{w}_{r}^\top \textbf{e}_t \textbf{w}_r

    where $\textbf{h}, \textbf{t} \in \mathbb{R}^d$. Then, the projected embeddings are used to compute the score
    for the triple $(h,r,t)$:

    .. math::

        f(h, r, t) = -\|\textbf{e'}_{h,r} + \textbf{d}_r - \textbf{e'}_{t,r}\|_{p}^2

    .. seealso::

       - OpenKE `implementation of TransH <https://github.com/thunlp/OpenKE/blob/master/models/TransH.py>`_
    ---
    citation:
        author: Wang
        year: 2014
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )
    #: The custom regularizer used by [wang2014]_ for TransH
    regularizer_default: ClassVar[Type[Regularizer]] = NormLimitRegularizer
    #: The settings used by [wang2014]_ for TransH
    # The regularization in TransH enforces the defined soft constraints that should computed only for every batch.
    # Therefore, apply_only_once is always set to True.
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.05, apply_only_once=True, dim=-1, p=2, power_norm=True, max_norm=1.0
    )
    #: The custom regularizer used by [wang2014]_ for TransH
    relation_regularizer_default: ClassVar[Type[Regularizer]] = OrthogonalityRegularizer
    #: The settings used by [wang2014]_ for TransH
    relation_regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.05, apply_only_once=True, epsilon=1e-5
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 2,
        entity_initializer: Hint[Initializer] = init.xavier_normal_,
        entity_regularizer: HintOrType[Regularizer] = None,
        entity_regularizer_kwargs: OptionalKwargs = None,
        relation_initializer: Hint[Initializer] = init.xavier_normal_,
        relation_regularizer: HintOrType[Regularizer] = None,
        relation_regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        r"""Initialize TransH.

        :param embedding_dim:
            the entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm:
            the :math:`l_p` norm applied in the interaction function. Is usually ``1`` or ``2.``.

        :param entity_initializer:
            the entity initializer function
        :param entity_regularizer:
            the entity regularizer. Defaults to :attr:`pykeen.models.TransH.regularizer_default`
        :param entity_regularizer_kwargs:
            keyword-based parameters for the entity regularizer. If `entity_regularizer` is None,
            the default from :attr:`pykeen.models.TransH.regularizer_default_kwargs` will be used instead

        :param relation_initializer:
            relation initializer function
        :param relation_regularizer:
            the relation regularizer. Defaults to :attr:`pykeen.models.TransH.relation_regularizer_default`
        :param relation_regularizer_kwargs:
            keyword-based parameters for the relation regularizer. If `relation_regularizer` is None,
            the default from :attr:`pykeen.models.TransH.relation_regularizer_default_kwargs` will be used instead

        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.ERModel`
        """
        super().__init__(
            interaction=TransHInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=[
                # translation vector in hyperplane
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                ),
                # normal vector of hyperplane
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    # normalise the normal vectors to unit l2 length
                    constrainer=functional.normalize,
                ),
            ],
            **kwargs,
        )

        # As described in [wang2014], all entities and relations are used to compute the regularization term
        # which enforces the defined soft constraints.
        # thus, we need to use a weight regularizer instead of having an Embedding regularizer,
        # which only regularizes the weights used in a batch
        self.append_weight_regularizer(
            parameter=self.entity_representations[0].parameters(),
            regularizer=entity_regularizer,
            regularizer_kwargs=entity_regularizer_kwargs,
            # note: the following is already the default
            # default_regularizer=self.regularizer_default,
            # default_regularizer_kwargs=self.regularizer_default_kwargs,
        )
        self.append_weight_regularizer(
            parameter=itertools.chain.from_iterable(r.parameters() for r in self.relation_representations),
            regularizer=relation_regularizer,
            regularizer_kwargs=relation_regularizer_kwargs,
            default_regularizer=self.relation_regularizer_default,
            default_regularizer_kwargs=self.relation_regularizer_default_kwargs,
        )
