# -*- coding: utf-8 -*-

"""Implementation of the TransE model."""

from typing import Optional

import torch
import torch.autograd
from torch.nn import functional

from ..base import EntityRelationEmbeddingModel
from ..init import embedding_xavier_uniform_
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory

__all__ = [
    'TransE',
]


class TransE(EntityRelationEmbeddingModel):
    """An implementation of TransE from [bordes2013]_.

     This model considers a relation as a translation from the head to the tail entity.

    .. seealso::

       - OpenKE `implementation of TransE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        scoring_fct_norm=dict(type=int, low=1, high=2),
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

        # Finalize initialization
        self.reset_parameters_()

    def _reset_parameters_(self):  # noqa: D102
        embedding_xavier_uniform_(self.entity_embeddings)
        embedding_xavier_uniform_(self.relation_embeddings)
        # Initialise relation embeddings to unit length
        functional.normalize(self.relation_embeddings.weight.data, out=self.relation_embeddings.weight.data)

    def post_parameter_update(self) -> None:  # noqa: D102
        # Make sure to call super first
        super().post_parameter_update()

        # Normalize entity embeddings
        functional.normalize(self.entity_embeddings.weight.data, out=self.entity_embeddings.weight.data)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hrt_batch[:, 0])
        r = self.relation_embeddings(hrt_batch[:, 1])
        t = self.entity_embeddings(hrt_batch[:, 2])

        return -torch.norm(h + r - t, dim=-1, p=self.scoring_fct_norm, keepdim=True)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hr_batch[:, 0])
        r = self.relation_embeddings(hr_batch[:, 1])
        t = self.entity_embeddings.weight

        return -torch.norm(h[:, None, :] + r[:, None, :] - t[None, :, :], dim=-1, p=self.scoring_fct_norm)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(rt_batch[:, 0])
        t = self.entity_embeddings(rt_batch[:, 1])

        return -torch.norm(h[None, :, :] + r[:, None, :] - t[:, None, :], dim=-1, p=self.scoring_fct_norm)
