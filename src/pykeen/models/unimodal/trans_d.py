# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from typing import Optional

import torch
import torch.autograd

from ..base import EntityRelationEmbeddingModel
from ..init import embedding_xavier_normal_
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...utils import clamp_norm, get_embedding, get_embedding_in_canonical_shape

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


class TransD(EntityRelationEmbeddingModel):
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
        preferred_device: Optional[str] = None,
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
        )

        self.entity_projections = get_embedding(
            num_embeddings=triples_factory.num_entities,
            embedding_dim=embedding_dim,
            device=self.device,
        )
        self.relation_projections = get_embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=relation_dim,
            device=self.device,
        )

        # Finalize initialization
        self.reset_parameters_()

    def post_parameter_update(self) -> None:  # noqa: D102
        # Make sure to call super first
        super().post_parameter_update()

        # Normalize entity embeddings
        self.entity_embeddings.weight.data = clamp_norm(x=self.entity_embeddings.weight.data, maxnorm=1., p=2, dim=-1)
        self.relation_embeddings.weight.data = clamp_norm(
            x=self.relation_embeddings.weight.data,
            maxnorm=1.,
            p=2,
            dim=-1,
        )

    def _reset_parameters_(self):  # noqa: D102
        embedding_xavier_normal_(self.entity_embeddings)
        embedding_xavier_normal_(self.entity_projections)
        embedding_xavier_normal_(self.relation_embeddings)
        embedding_xavier_normal_(self.relation_projections)

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        h_p: torch.FloatTensor,
        r: torch.FloatTensor,
        r_p: torch.FloatTensor,
        t: torch.FloatTensor,
        t_p: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function for given embeddings.

        The embeddings have to be in a broadcastable shape.

        :param h: shape: (batch_size, num_entities, d_e)
            Head embeddings.
        :param h_p: shape: (batch_size, num_entities, d_e)
            Head projections.
        :param r: shape: (batch_size, num_entities, d_r)
            Relation embeddings.
        :param r_p: shape: (batch_size, num_entities, d_r)
            Relation projections.
        :param t: shape: (batch_size, num_entities, d_e)
            Tail embeddings.
        :param t_p: shape: (batch_size, num_entities, d_e)
            Tail projections.

        :return: shape: (batch_size, num_entities)
            The scores.
        """
        # Project entities
        h_bot = _project_entity(e=h, e_p=h_p, r=r, r_p=r_p)
        t_bot = _project_entity(e=t, e_p=t_p, r=r, r_p=r_p)

        # score = -||h_bot + r - t_bot||_2^2
        return -torch.norm(h_bot + r - t_bot, dim=-1, p=2) ** 2

    def _score(
        self,
        h_ind: Optional[torch.LongTensor] = None,
        r_ind: Optional[torch.LongTensor] = None,
        t_ind: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Evaluate the interaction function.

        :param h_ind: shape: (batch_size,)
            The indices for head entities. If None, score against all.
        :param r_ind: shape: (batch_size,)
            The indices for relations. If None, score against all.
        :param t_ind: shape: (batch_size,)
            The indices for tail entities. If None, score against all.

        :return: The scores, shape: (batch_size, num_entities)
        """
        # Head
        h = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=h_ind)
        h_p = get_embedding_in_canonical_shape(embedding=self.entity_projections, ind=h_ind)

        r = get_embedding_in_canonical_shape(embedding=self.relation_embeddings, ind=r_ind)
        r_p = get_embedding_in_canonical_shape(embedding=self.relation_projections, ind=r_ind)

        t = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=t_ind)
        t_p = get_embedding_in_canonical_shape(embedding=self.entity_projections, ind=t_ind)

        return self.interaction_function(h=h, h_p=h_p, r=r, r_p=r_p, t=t, t_p=t_p)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hrt_batch[:, 0], r_ind=hrt_batch[:, 1], t_ind=hrt_batch[:, 2])

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hr_batch[:, 0], r_ind=hr_batch[:, 1], t_ind=None)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=None, r_ind=rt_batch[:, 0], t_ind=rt_batch[:, 1])
