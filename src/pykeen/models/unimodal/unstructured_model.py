# -*- coding: utf-8 -*-

"""Implementation of UM."""

from typing import Optional

import torch
import torch.autograd

from ..base import EntityEmbeddingModel
from ..init import embedding_xavier_uniform_
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory

__all__ = [
    'UnstructuredModel',
]


class UnstructuredModel(EntityEmbeddingModel):
    """An implementation of Unstructured Model (UM) from [bordes2014]_."""

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
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

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(hrt_batch[:, 0])
        t = self.entity_embeddings(hrt_batch[:, 2])

        return -torch.norm(h - t, dim=-1, p=self.scoring_fct_norm, keepdim=True) ** 2

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(hr_batch[:, 0]).view(-1, 1, self.embedding_dim)
        t = self.entity_embeddings.weight.view(1, -1, self.embedding_dim)

        return -torch.norm(h - t, dim=-1, p=self.scoring_fct_norm) ** 2

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings.weight.view(1, -1, self.embedding_dim)
        t = self.entity_embeddings(rt_batch[:, 1]).view(-1, 1, self.embedding_dim)

        return -torch.norm(h - t, dim=-1, p=self.scoring_fct_norm) ** 2
