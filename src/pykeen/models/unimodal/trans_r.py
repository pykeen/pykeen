# -*- coding: utf-8 -*-

"""Implementation of TransR."""

from functools import partial
from typing import Optional

import torch
import torch.autograd
import torch.nn.init
from torch.nn import functional

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...nn import Embedding
from ...nn.init import xavier_uniform_
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import clamp_norm, compose

__all__ = [
    'TransR',
]


def _projection_initializer(
    x: torch.FloatTensor,
    num_relations: int,
    embedding_dim: int,
    relation_dim: int,
) -> torch.FloatTensor:
    """Initialize by Glorot."""
    return torch.nn.init.xavier_uniform_(x.view(num_relations, embedding_dim, relation_dim)).view(x.shape)


class TransR(EntityRelationEmbeddingModel):
    r"""An implementation of TransR from [lin2015]_.

    TransR is an extension of :class:`pykeen.models.TransH` that explicitly considers entities and relations as
    different objects and therefore represents them in different vector spaces.

    For a triple $(h,r,t) \in \mathbb{K}$, the entity embeddings, $\textbf{e}_h, \textbf{e}_t \in \mathbb{R}^d$,
    are first projected into the relation space by means of a relation-specific projection matrix
    $\textbf{M}_{r} \in \mathbb{R}^{k \times d}$. With relation embedding $\textbf{r}_r \in \mathbb{R}^k$, the
    interaction model is defined similarly to TransE with:

    .. math::

        f(h,r,t) = -\|\textbf{M}_{r}\textbf{e}_h + \textbf{r}_r - \textbf{M}_{r}\textbf{e}_t\|_{p}^2

    The following constraints are applied:

     * $\|\textbf{e}_h\|_2 \leq 1$
     * $\|\textbf{r}_r\|_2 \leq 1$
     * $\|\textbf{e}_t\|_2 \leq 1$
     * $\|\textbf{M}_{r}\textbf{e}_h\|_2 \leq 1$
     * $\|\textbf{M}_{r}\textbf{e}_t\|_2 \leq 1$

    .. seealso::

       - OpenKE `TensorFlow implementation of TransR
         <https://github.com/thunlp/OpenKE/blob/master/models/TransR.py>`_
       - OpenKE `PyTorch implementation of TransR
         <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransR.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=20, high=300, q=50),
        relation_dim=dict(type=int, low=20, high=300, q=50),
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        relation_dim: int = 30,
        scoring_fct_norm: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            relation_dim=relation_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            entity_initializer=xavier_uniform_,
            entity_constrainer=clamp_norm,
            entity_constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
            relation_initializer=compose(
                xavier_uniform_,
                functional.normalize,
            ),
            relation_constrainer=clamp_norm,
            relation_constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
        )
        self.scoring_fct_norm = scoring_fct_norm

        # TODO: Initialize from TransE

        # embeddings
        self.relation_projections = Embedding.init_with_device(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=relation_dim * embedding_dim,
            device=self.device,
            initializer=partial(
                _projection_initializer,
                num_relations=self.num_relations,
                embedding_dim=self.embedding_dim,
                relation_dim=self.relation_dim,
            ),
        )

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        self.relation_projections.reset_parameters()

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        m_r: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function for given embeddings.

        The embeddings have to be in a broadcastable shape.

        :param h: shape: (batch_size, num_entities, d_e)
            Head embeddings.
        :param r: shape: (batch_size, num_entities, d_r)
            Relation embeddings.
        :param t: shape: (batch_size, num_entities, d_e)
            Tail embeddings.
        :param m_r: shape: (batch_size, num_entities, d_e, d_r)
            The relation specific linear transformations.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        # project to relation specific subspace, shape: (b, e, d_r)
        h_bot = h @ m_r
        t_bot = t @ m_r
        # ensure constraints
        h_bot = clamp_norm(h_bot, p=2, dim=-1, maxnorm=1.)
        t_bot = clamp_norm(t_bot, p=2, dim=-1, maxnorm=1.)

        # evaluate score function, shape: (b, e)
        return -torch.norm(h_bot + r - t_bot, dim=-1) ** 2

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(indices=hrt_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings(indices=hrt_batch[:, 2]).unsqueeze(dim=1)
        m_r = self.relation_projections(indices=hrt_batch[:, 1]).view(-1, self.embedding_dim, self.relation_dim)

        return self.interaction_function(h=h, r=r, t=t, m_r=m_r).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(indices=hr_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings(indices=None).unsqueeze(dim=0)
        m_r = self.relation_projections(indices=hr_batch[:, 1]).view(-1, self.embedding_dim, self.relation_dim)

        return self.interaction_function(h=h, r=r, t=t, m_r=m_r)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=None).unsqueeze(dim=0)
        r = self.relation_embeddings(indices=rt_batch[:, 0]).unsqueeze(dim=1)
        t = self.entity_embeddings(indices=rt_batch[:, 1]).unsqueeze(dim=1)
        m_r = self.relation_projections(indices=rt_batch[:, 0]).view(-1, self.embedding_dim, self.relation_dim)

        return self.interaction_function(h=h, r=r, t=t, m_r=m_r)
