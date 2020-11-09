# -*- coding: utf-8 -*-

"""An implementation of TransH."""

from typing import Optional

import torch
from torch.nn import functional

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...nn import Embedding
from ...regularizers import Regularizer, TransHRegularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'TransH',
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
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )
    #: The custom regularizer used by [wang2014]_ for TransH
    regularizer_default = TransHRegularizer
    #: The settings used by [wang2014]_ for TransH
    regularizer_default_kwargs = dict(
        weight=0.05,
        epsilon=1e-5,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        scoring_fct_norm: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize TransH.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The :math:`l_p` norm applied in the interaction function. Is usually ``1`` or ``2.``.
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

        self.scoring_fct_norm = scoring_fct_norm

        # embeddings
        self.normal_vector_embeddings = Embedding.init_with_device(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=embedding_dim,
            device=self.device,
            # Normalise the normal vectors by their l2 norms
            constrainer=functional.normalize,
        )

    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()
        self.normal_vector_embeddings.post_parameter_update()

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        self.normal_vector_embeddings.reset_parameters()
        # TODO: Add initialization

    def regularize_if_necessary(self) -> None:
        """Update the regularizer's term given some tensors, if regularization is requested."""
        # As described in [wang2014], all entities and relations are used to compute the regularization term
        # which enforces the defined soft constraints.
        super().regularize_if_necessary(
            self.entity_embeddings(indices=None),
            self.normal_vector_embeddings(indices=None),  # FIXME
            self.relation_embeddings(indices=None),
        )

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
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

        return -torch.norm(ph + d_r - pt, p=2, dim=-1, keepdim=True)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
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

        return -torch.norm(ph[:, None, :] + d_r[:, None, :] - pt, p=2, dim=-1)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
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

        return -torch.norm(ph + d_r[:, None, :] - pt[:, None, :], p=2, dim=-1)
