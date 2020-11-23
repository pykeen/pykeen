# -*- coding: utf-8 -*-

"""Implementation of structured model (SE)."""

import functools
from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from ..base import EntityEmbeddingModel
from ...losses import Loss
from ...nn import Embedding
from ...nn.init import xavier_uniform_
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import compose

__all__ = [
    'StructuredEmbedding',
]


class StructuredEmbedding(EntityEmbeddingModel):
    r"""An implementation of the Structured Embedding (SE) published by [bordes2011]_.

    SE applies role- and relation-specific projection matrices
    $\textbf{M}_{r}^{h}, \textbf{M}_{r}^{t} \in \mathbb{R}^{d \times d}$ to the head and tail
    entities' embeddings before computing their differences. Then, the $l_p$ norm is applied
    and the result is negated such that smaller differences are considered better.

    .. math::

        f(h, r, t) = - \|\textbf{M}_{r}^{h} \textbf{e}_h  - \textbf{M}_{r}^{t} \textbf{e}_t\|_p

    By employing different projections for the embeddings of the head and tail entities, SE explicitly differentiates
    the role of an entity as either the subject or object.
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
        scoring_fct_norm: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize SE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The $l_p$ norm. Usually 1 for SE.
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            entity_initializer=xavier_uniform_,
            entity_constrainer=functional.normalize,
        )

        self.scoring_fct_norm = scoring_fct_norm

        # Embeddings
        init_bound = 6 / np.sqrt(self.embedding_dim)
        # Initialise relation embeddings to unit length
        initializer = compose(
            functools.partial(nn.init.uniform_, a=-init_bound, b=+init_bound),
            functional.normalize,
        )
        self.left_relation_embeddings = Embedding.init_with_device(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=embedding_dim ** 2,
            device=self.device,
            initializer=initializer,
        )
        self.right_relation_embeddings = Embedding.init_with_device(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=embedding_dim ** 2,
            device=self.device,
            initializer=initializer,
        )

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        self.left_relation_embeddings.reset_parameters()
        self.right_relation_embeddings.reset_parameters()

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0]).view(-1, self.embedding_dim, 1)
        rel_h = self.left_relation_embeddings(indices=hrt_batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        rel_t = self.right_relation_embeddings(indices=hrt_batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        t = self.entity_embeddings(indices=hrt_batch[:, 2]).view(-1, self.embedding_dim, 1)

        # Project entities
        proj_h = rel_h @ h
        proj_t = rel_t @ t

        scores = -torch.norm(proj_h - proj_t, dim=1, p=self.scoring_fct_norm)
        return scores

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0]).view(-1, self.embedding_dim, 1)
        rel_h = self.left_relation_embeddings(indices=hr_batch[:, 1]).view(-1, self.embedding_dim, self.embedding_dim)
        rel_t = self.right_relation_embeddings(indices=hr_batch[:, 1])
        rel_t = rel_t.view(-1, 1, self.embedding_dim, self.embedding_dim)
        t_all = self.entity_embeddings(indices=None).view(1, -1, self.embedding_dim, 1)

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
        h_all = self.entity_embeddings(indices=None).view(1, -1, self.embedding_dim, 1)
        rel_h = self.left_relation_embeddings(indices=rt_batch[:, 0])
        rel_h = rel_h.view(-1, 1, self.embedding_dim, self.embedding_dim)
        rel_t = self.right_relation_embeddings(indices=rt_batch[:, 0]).view(-1, self.embedding_dim, self.embedding_dim)
        t = self.entity_embeddings(indices=rt_batch[:, 1]).view(-1, self.embedding_dim, 1)

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
