# -*- coding: utf-8 -*-

"""Implementation of structured model (SE)."""

from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from ..base import Model
from ..init import embedding_xavier_uniform_
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory

__all__ = [
    'StructuredEmbedding',
]


class StructuredEmbedding(Model):
    """An implementation of Structured Embedding (SE) from [bordes2011]_.

    This model projects different matrices for each relation head and tail entity.
    """

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
        left_relation_embeddings: Optional[nn.Embedding] = None,
        right_relation_embeddings: Optional[nn.Embedding] = None,
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

        # Embeddings
        self.scoring_fct_norm = scoring_fct_norm

        self.left_relation_embeddings = left_relation_embeddings
        self.right_relation_embeddings = right_relation_embeddings

        # Finalize initialization
        self._init_weights_on_device()

    def init_empty_weights_(self):  # noqa: D102
        init_bound = 6 / np.sqrt(self.embedding_dim)
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_uniform_(self.entity_embeddings)
        if self.left_relation_embeddings is None:
            self.left_relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim ** 2)
            nn.init.uniform_(
                self.left_relation_embeddings.weight.data,
                a=-init_bound,
                b=+init_bound,
            )
            # Initialise left relation embeddings to unit length
            functional.normalize(
                self.left_relation_embeddings.weight.data,
                out=self.left_relation_embeddings.weight.data,
            )
        if not self.right_relation_embeddings:
            self.right_relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim ** 2)
            nn.init.uniform_(
                self.right_relation_embeddings.weight.data,
                a=-init_bound,
                b=+init_bound,
            )
            # Initialise right relation embeddings to unit length
            functional.normalize(
                self.right_relation_embeddings.weight.data,
                out=self.right_relation_embeddings.weight.data,
            )
        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.left_relation_embeddings = None
        self.right_relation_embeddings = None
        return self

    def post_parameter_update(self) -> None:  # noqa: D102
        # Make sure to call super first
        super().post_parameter_update()

        # Normalise embeddings of entities
        functional.normalize(self.entity_embeddings.weight.data, out=self.entity_embeddings.weight.data)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hrt_batch[:, 0]).view(-1, self.embedding_dim, 1)
        rel_h = self.left_relation_embeddings(hrt_batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        rel_t = self.right_relation_embeddings(hrt_batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        t = self.entity_embeddings(hrt_batch[:, 2]).view(-1, self.embedding_dim, 1)

        # Project entities
        proj_h = rel_h @ h
        proj_t = rel_t @ t

        scores = -torch.norm(proj_h - proj_t, dim=1, p=self.scoring_fct_norm)
        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hr_batch[:, 0]).view(-1, self.embedding_dim, 1)
        rel_h = self.left_relation_embeddings(hr_batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        rel_t = self.right_relation_embeddings(hr_batch[:, 1]).view(-1, 1, self.embedding_dim, self.embedding_dim)
        t_all = self.entity_embeddings.weight.view(1, -1, self.embedding_dim, 1)

        if slice_size is not None:
            proj_t_arr = []
            # Project entities
            proj_h = rel_h @ h

            for t in torch.split(t_all, slice_size, dim=1):
                # Project entities
                proj_t = rel_t @ t
                proj_t_arr.append(proj_t)

            proj_t = torch.cat(proj_t_arr, dim=1)

        else:
            # Project entities
            proj_h = rel_h @ h
            proj_t = rel_t @ t_all

        scores = -torch.norm(proj_h[:, None, :, 0] - proj_t[:, :, :, 0], dim=-1, p=self.scoring_fct_norm)

        return scores

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h_all = self.entity_embeddings.weight.view(1, -1, self.embedding_dim, 1)
        rel_h = self.left_relation_embeddings(rt_batch[:, 0]).view(-1, 1, self.embedding_dim, self.embedding_dim)
        rel_t = self.right_relation_embeddings(rt_batch[:, 0]).view(-1, self.embedding_dim, self.embedding_dim)
        t = self.entity_embeddings(rt_batch[:, 1]).view(-1, self.embedding_dim, 1)

        if slice_size is not None:
            proj_h_arr = []

            # Project entities
            proj_t = rel_t @ t

            for h in torch.split(h_all, slice_size, dim=1):
                # Project entities
                proj_h = rel_h @ h
                proj_h_arr.append(proj_h)

            proj_h = torch.cat(proj_h_arr, dim=1)
        else:
            # Project entities
            proj_h = rel_h @ h_all
            proj_t = rel_t @ t

        scores = -torch.norm(proj_h[:, :, :, 0] - proj_t[:, None, :, 0], dim=-1, p=self.scoring_fct_norm)

        return scores
