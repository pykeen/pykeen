# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from typing import Optional

import torch
import torch.autograd

from ..base import TwoVectorEmbeddingModel
from ...losses import Loss
from ...nn.emb import EmbeddingSpecification
from ...nn.init import xavier_normal_
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import clamp_norm

__all__ = [
    'TransD',
]


def _project_entity(
    e: torch.FloatTensor,
    e_p: torch.FloatTensor,
    r: torch.FloatTensor,
    r_p: torch.FloatTensor,
) -> torch.FloatTensor:
    r"""Project entity relation-specific.

    .. math::

        e_{\bot} = M_{re} e
                 = (r_p e_p^T + I^{d_r \times d_e}) e
                 = r_p e_p^T e + I^{d_r \times d_e} e
                 = r_p (e_p^T e) + e'

    and additionally enforces

    .. math::

        \|e_{\bot}\|_2 \leq 1

    :param e: shape: (batch_size, num_entities, d_e)
        The entity embedding.
    :param e_p: shape: (batch_size, num_entities, d_e)
        The entity projection.
    :param r: shape: (batch_size, num_entities, d_r)
        The relation embedding.
    :param r_p: shape: (batch_size, num_entities, d_r)
        The relation projection.

    :return: shape: (batch_size, num_entities, d_r)

    """
    # The dimensions affected by e'
    change_dim = min(e.shape[-1], r.shape[-1])

    # Project entities
    # r_p (e_p.T e) + e'
    e_bot = r_p * torch.sum(e_p * e, dim=-1, keepdim=True)
    e_bot[:, :, :change_dim] += e[:, :, :change_dim]

    # Enforce constraints
    e_bot = clamp_norm(e_bot, p=2, dim=-1, maxnorm=1)

    return e_bot


class TransD(TwoVectorEmbeddingModel):
    r"""An implementation of TransD from [ji2015]_.

    TransD is an extension of :class:`pykeen.models.TransR` that, like TransR, considers entities and relations
    as objects living in different vector spaces. However, instead of performing the same relation-specific
    projection for all entity embeddings, entity-relation-specific projection matrices
    $\textbf{M}_{r,h}, \textbf{M}_{t,h}  \in \mathbb{R}^{k \times d}$ are constructed.

    To do so, all head entities, tail entities, and relations are represented by two vectors,
    $\textbf{e}_h, \hat{\textbf{e}}_h, \textbf{e}_t, \hat{\textbf{e}}_t \in \mathbb{R}^d$ and
    $\textbf{r}_r, \hat{\textbf{r}}_r \in \mathbb{R}^k$, respectively. The first set of embeddings is used for
    calculating the entity-relation-specific projection matrices:

    .. math::

        \textbf{M}_{r,h} = \hat{\textbf{r}}_r \hat{\textbf{e}}_h^{T} + \tilde{\textbf{I}}

        \textbf{M}_{r,t} = \hat{\textbf{r}}_r \hat{\textbf{e}}_t^{T} + \tilde{\textbf{I}}

    where $\tilde{\textbf{I}} \in \mathbb{R}^{k \times d}$ is a $k \times d$ matrix with ones on the diagonal and
    zeros elsewhere. Next, $\textbf{e}_h$ and $\textbf{e}_t$ are projected into the relation space by means of the
    constructed projection matrices. Finally, the plausibility score for $(h,r,t) \in \mathbb{K}$ is given by:

    .. math::

        f(h,r,t) = -\|\textbf{M}_{r,h} \textbf{e}_h + \textbf{r}_r - \textbf{M}_{r,t} \textbf{e}_t\|_{2}^2

    .. seealso::

       - OpenKE `implementation of TransD <https://github.com/thunlp/OpenKE/blob/master/models/TransD.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=20, high=300, q=50),
        relation_dim=dict(type=int, low=20, high=300, q=50),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        relation_dim: int = 30,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            relation_dim=relation_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            embedding_specification=EmbeddingSpecification(
                initializer=xavier_normal_,
                constrainer=clamp_norm,
                constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
            ),
            relation_embedding_specification=EmbeddingSpecification(
                initializer=xavier_normal_,
                constrainer=clamp_norm,
                constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
            ),
        )

    def _forward(
        self,
        h1: torch.FloatTensor,
        h2: torch.FloatTensor,
        r1: torch.FloatTensor,
        r2: torch.FloatTensor,
        t1: torch.FloatTensor,
        t2: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa:D102
        # Project entities
        h_bot = _project_entity(e=h1, e_p=h2, r=r1, r_p=r2)
        t_bot = _project_entity(e=t1, e_p=t2, r=r1, r_p=r2)
        return self.interaction_function(h=h_bot, r=r1, t=t_bot)
