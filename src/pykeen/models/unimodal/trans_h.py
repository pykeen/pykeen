# -*- coding: utf-8 -*-

"""An implementation of TransH."""

from typing import Any, ClassVar, Mapping, Type

import torch
from torch import linalg
from torch.nn import functional
from torch.nn.init import uniform_

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import representation_resolver
from ...regularizers import Regularizer, TransHRegularizer
from ...typing import Hint, Initializer

__all__ = [
    "TransH",
]


class TransH(EntityRelationEmbeddingModel):
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
    regularizer_default: ClassVar[Type[Regularizer]] = TransHRegularizer
    #: The settings used by [wang2014]_ for TransH
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.05,
        epsilon=1e-5,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 2,
        entity_initializer: Hint[Initializer] = uniform_,
        relation_initializer: Hint[Initializer] = uniform_,
        **kwargs,
    ) -> None:
        r"""Initialize TransH.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The :math:`l_p` norm applied in the interaction function. Is usually ``1`` or ``2.``.
        :param entity_initializer: Entity initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param relation_initializer: Relation initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.EntityRelationEmbeddingModel`
        """
        super().__init__(
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
        )

        self.scoring_fct_norm = scoring_fct_norm

        # embeddings
        self.normal_vector_embeddings = representation_resolver.make(
            query=None,
            max_id=self.num_relations,
            shape=embedding_dim,
            # Normalise the normal vectors by their l2 norms
            constrainer=functional.normalize,
        )

    # docstr-coverage: inherited
    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()
        self.normal_vector_embeddings.post_parameter_update()

    # docstr-coverage: inherited
    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        self.normal_vector_embeddings.reset_parameters()
        # TODO: Add initialization

    def regularize_if_necessary(self, *tensors: torch.FloatTensor) -> None:
        """Update the regularizer's term given some tensors, if regularization is requested."""
        if tensors:
            raise ValueError("no tensors should be passed to TransH.regularize_if_necessary")
        # As described in [wang2014], all entities and relations are used to compute the regularization term
        # which enforces the defined soft constraints.
        super().regularize_if_necessary(
            self.entity_embeddings(indices=None),
            self.normal_vector_embeddings(indices=None),  # FIXME
            self.relation_embeddings(indices=None),
        )

    # docstr-coverage: inherited
    def score_hrt(self, hrt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0])
        d_r = self.relation_embeddings(indices=hrt_batch[:, 1])
        w_r = self.normal_vector_embeddings(indices=hrt_batch[:, 1])
        t = self.entity_embeddings(indices=hrt_batch[:, 2])

        # Project to hyperplane
        ph = h - torch.sum(w_r * h, dim=-1, keepdim=True) * w_r
        pt = t - torch.sum(w_r * t, dim=-1, keepdim=True) * w_r

        # Regularization term
        self.regularize_if_necessary()

        return -linalg.vector_norm(ph + d_r - pt, ord=2, dim=-1, keepdim=True)

    # docstr-coverage: inherited
    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0])
        d_r = self.relation_embeddings(indices=hr_batch[:, 1])
        w_r = self.normal_vector_embeddings(indices=hr_batch[:, 1])
        t = self.entity_embeddings(indices=None)

        # Project to hyperplane
        ph = h - torch.sum(w_r * h, dim=-1, keepdim=True) * w_r
        pt = t[None, :, :] - torch.sum(w_r[:, None, :] * t[None, :, :], dim=-1, keepdim=True) * w_r[:, None, :]

        # Regularization term
        self.regularize_if_necessary()

        return -linalg.vector_norm(ph[:, None, :] + d_r[:, None, :] - pt, ord=2, dim=-1)

    # docstr-coverage: inherited
    def score_h(self, rt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=None)
        rel_id = rt_batch[:, 0]
        d_r = self.relation_embeddings(indices=rel_id)
        w_r = self.normal_vector_embeddings(indices=rel_id)
        t = self.entity_embeddings(indices=rt_batch[:, 1])

        # Project to hyperplane
        ph = h[None, :, :] - torch.sum(w_r[:, None, :] * h[None, :, :], dim=-1, keepdim=True) * w_r[:, None, :]
        pt = t - torch.sum(w_r * t, dim=-1, keepdim=True) * w_r

        # Regularization term
        self.regularize_if_necessary()

        return -linalg.vector_norm(ph + (d_r[:, None, :] - pt[:, None, :]), ord=2, dim=-1)
