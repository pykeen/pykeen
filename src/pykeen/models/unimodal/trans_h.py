# -*- coding: utf-8 -*-

"""An implementation of TransH."""

from typing import Optional

import torch
from torch.nn import functional

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...regularizers import Regularizer, TransHRegularizer
from ...triples import TriplesFactory
from ...utils import get_embedding

__all__ = [
    'TransH',
]


class TransH(EntityRelationEmbeddingModel):
    """An implementation of TransH [wang2014]_.

    This model extends TransE by applying the translation from head to tail entity in a relational-specific hyperplane.

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
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
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
        self.normal_vector_embeddings = get_embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=embedding_dim,
            device=self.device,
        )

        # Finalize initialization
        self.reset_parameters_()

    def _reset_parameters_(self):  # noqa: D102
        for emb in [
            self.entity_embeddings,
            self.relation_embeddings,
            self.normal_vector_embeddings,
        ]:
            emb.reset_parameters()
        # TODO: Add initialization

    def post_parameter_update(self) -> None:  # noqa: D102
        # Make sure to call super first
        super().post_parameter_update()

        # Normalise the normal vectors by their l2 norms
        functional.normalize(
            self.normal_vector_embeddings.weight.data,
            out=self.normal_vector_embeddings.weight.data,
        )

    def regularize_if_necessary(self) -> None:
        """Update the regularizer's term given some tensors, if regularization is requested."""
        # As described in [wang2014], all entities and relations are used to compute the regularization term
        # which enforces the defined soft constraints.
        super().regularize_if_necessary(
            self.entity_embeddings.weight,
            self.normal_vector_embeddings.weight,
            self.relation_embeddings.weight,
        )

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hrt_batch[:, 0])
        d_r = self.relation_embeddings(hrt_batch[:, 1])
        w_r = self.normal_vector_embeddings(hrt_batch[:, 1])
        t = self.entity_embeddings(hrt_batch[:, 2])

        # Project to hyperplane
        ph = h - torch.sum(w_r * h, dim=-1, keepdim=True) * w_r
        pt = t - torch.sum(w_r * t, dim=-1, keepdim=True) * w_r

        # Regularization term
        self.regularize_if_necessary()

        return -torch.norm(ph + d_r - pt, p=2, dim=-1, keepdim=True)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hr_batch[:, 0])
        d_r = self.relation_embeddings(hr_batch[:, 1])
        w_r = self.normal_vector_embeddings(hr_batch[:, 1])
        t = self.entity_embeddings.weight

        # Project to hyperplane
        ph = h - torch.sum(w_r * h, dim=-1, keepdim=True) * w_r
        pt = t[None, :, :] - torch.sum(w_r[:, None, :] * t[None, :, :], dim=-1, keepdim=True) * w_r[:, None, :]

        # Regularization term
        self.regularize_if_necessary()

        return -torch.norm(ph[:, None, :] + d_r[:, None, :] - pt, p=2, dim=-1)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings.weight
        rel_id = rt_batch[:, 0]
        d_r = self.relation_embeddings(rel_id)
        w_r = self.normal_vector_embeddings(rel_id)
        t = self.entity_embeddings(rt_batch[:, 1])

        # Project to hyperplane
        ph = h[None, :, :] - torch.sum(w_r[:, None, :] * h[None, :, :], dim=-1, keepdim=True) * w_r[:, None, :]
        pt = t - torch.sum(w_r * t, dim=-1, keepdim=True) * w_r

        # Regularization term
        self.regularize_if_necessary()

        return -torch.norm(ph + d_r[:, None, :] - pt[:, None, :], p=2, dim=-1)
