# -*- coding: utf-8 -*-

"""Implementation of ProjE."""

from typing import Optional

import numpy
import torch
import torch.autograd
from torch import nn

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...nn.init import xavier_uniform_
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ProjE',
]


class ProjE(EntityRelationEmbeddingModel):
    r"""An implementation of ProjE from [shi2017]_.

    ProjE is a neural network-based approach with a *combination* and a *projection* layer. The interaction model
    first combines $h$ and $r$ by following combination operator:

    .. math::

        \textbf{h} \otimes \textbf{r} = \textbf{D}_e \textbf{h} + \textbf{D}_r \textbf{r} + \textbf{b}_c

    where $\textbf{D}_e, \textbf{D}_r \in \mathbb{R}^{k \times k}$ are diagonal matrices which are used as shared
    parameters among all entities and relations, and $\textbf{b}_c \in \mathbb{R}^{k}$ represents the candidate bias
    vector shared across all entities. Next, the score for the triple $(h,r,t) \in \mathbb{K}$ is computed:

    .. math::

        f(h, r, t) = g(\textbf{t} \ z(\textbf{h} \otimes \textbf{r}) + \textbf{b}_p)

    where $g$ and $z$ are activation functions, and $\textbf{b}_p$ represents the shared projection bias vector.

    .. seealso::

       - Official Implementation: https://github.com/nddsg/ProjE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )
    #: The default loss function class
    loss_default = nn.BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(reduction='mean')

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        inner_non_linearity: Optional[nn.Module] = None,
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
            entity_initializer=xavier_uniform_,
            relation_initializer=xavier_uniform_,
        )

        # Global entity projection
        self.d_e = nn.Parameter(torch.empty(self.embedding_dim, device=self.device), requires_grad=True)

        # Global relation projection
        self.d_r = nn.Parameter(torch.empty(self.embedding_dim, device=self.device), requires_grad=True)

        # Global combination bias
        self.b_c = nn.Parameter(torch.empty(self.embedding_dim, device=self.device), requires_grad=True)

        # Global combination bias
        self.b_p = nn.Parameter(torch.empty(1, device=self.device), requires_grad=True)

        if inner_non_linearity is None:
            inner_non_linearity = nn.Tanh()
        self.inner_non_linearity = inner_non_linearity

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        bound = numpy.sqrt(6) / self.embedding_dim
        nn.init.uniform_(self.d_e, a=-bound, b=bound)
        nn.init.uniform_(self.d_r, a=-bound, b=bound)
        nn.init.uniform_(self.b_c, a=-bound, b=bound)
        nn.init.uniform_(self.b_p, a=-bound, b=bound)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0])
        r = self.relation_embeddings(indices=hrt_batch[:, 1])
        t = self.entity_embeddings(indices=hrt_batch[:, 2])

        # Compute score
        hidden = self.inner_non_linearity(self.d_e[None, :] * h + self.d_r[None, :] * r + self.b_c[None, :])
        scores = torch.sum(hidden * t, dim=-1, keepdim=True) + self.b_p

        return scores

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0])
        r = self.relation_embeddings(indices=hr_batch[:, 1])
        t = self.entity_embeddings(indices=None)

        # Rank against all entities
        hidden = self.inner_non_linearity(self.d_e[None, :] * h + self.d_r[None, :] * r + self.b_c[None, :])
        scores = torch.sum(hidden[:, None, :] * t[None, :, :], dim=-1) + self.b_p

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=None)
        r = self.relation_embeddings(indices=rt_batch[:, 0])
        t = self.entity_embeddings(indices=rt_batch[:, 1])

        # Rank against all entities
        hidden = self.inner_non_linearity(
            self.d_e[None, None, :] * h[None, :, :]
            + (self.d_r[None, None, :] * r[:, None, :] + self.b_c[None, None, :]),
        )
        scores = torch.sum(hidden * t[:, None, :], dim=-1) + self.b_p

        return scores
