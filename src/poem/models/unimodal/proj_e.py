# -*- coding: utf-8 -*-

"""Implementation of ProjE."""

from typing import Optional

import numpy
import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ..init import embedding_xavier_uniform_
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory

__all__ = [
    'ProjE',
]


class ProjE(BaseModule):
    """An implementation of ProjE from [shi2017]_.

    .. seealso::

       - Official Implementation: <https://github.com/nddsg/ProjE>`_
    """

    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )

    loss_default = nn.BCEWithLogitsLoss
    loss_default_kwargs = dict(reduction='mean')

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        entity_embeddings: Optional[nn.Embedding] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        loss: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        inner_non_linearity: Optional[nn.Module] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        if inner_non_linearity is None:
            inner_non_linearity = nn.Tanh()

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.relation_embeddings = relation_embeddings
        self.inner_non_linearity = inner_non_linearity

        # Global entity projection
        self.d_e = nn.Parameter(torch.empty(self.embedding_dim, requires_grad=True))

        # Global relation projection
        self.d_r = nn.Parameter(torch.empty(self.embedding_dim, requires_grad=True))

        # Global combination bias
        self.b_c = nn.Parameter(torch.empty(self.embedding_dim, requires_grad=True))

        # Global combination bias
        self.b_p = nn.Parameter(torch.empty(1, requires_grad=True))

        # Finalize initialization
        self._init_weights_on_device()

    def init_empty_weights_(self):  # noqa: D102
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_uniform_(self.entity_embeddings)

        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
            embedding_xavier_uniform_(self.relation_embeddings)

        # TODO: How to determine whether weights have been initialized
        bound = numpy.sqrt(6) / self.embedding_dim
        nn.init.uniform_(self.d_e, a=-bound, b=bound)
        nn.init.uniform_(self.d_r, a=-bound, b=bound)
        nn.init.uniform_(self.b_c, a=-bound, b=bound)
        nn.init.uniform_(self.b_p, a=-bound, b=bound)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.relation_embeddings = None
        return self

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hrt_batch[:, 0])
        r = self.relation_embeddings(hrt_batch[:, 1])
        t = self.entity_embeddings(hrt_batch[:, 2])

        # Compute score
        hidden = self.inner_non_linearity(self.d_e[None, :] * h + self.d_r[None, :] * r + self.b_c[None, :])
        scores = torch.sum(hidden * t, dim=-1, keepdim=True) + self.b_p

        return scores

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hr_batch[:, 0])
        r = self.relation_embeddings(hr_batch[:, 1])
        t = self.entity_embeddings.weight

        # Rank against all entities
        hidden = self.inner_non_linearity(self.d_e[None, :] * h + self.d_r[None, :] * r + self.b_c[None, :])
        scores = torch.sum(hidden[:, None, :] * t[None, :, :], dim=-1) + self.b_p

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(rt_batch[:, 0])
        t = self.entity_embeddings(rt_batch[:, 1])

        # Rank against all entities
        hidden = self.inner_non_linearity(
            self.d_e[None, None, :] * h[None, :, :]
            + (self.d_r[None, None, :] * r[:, None, :] + self.b_c[None, None, :]),
        )
        scores = torch.sum(hidden * t[:, None, :], dim=-1) + self.b_p

        return scores
