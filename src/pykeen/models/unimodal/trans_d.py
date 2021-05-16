# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from typing import Any, ClassVar, Mapping, Optional

import torch
import torch.autograd

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import Embedding, EmbeddingSpecification
from ...nn.init import xavier_normal_, xavier_uniform_, xavier_uniform_norm_
from ...typing import Constrainer, Hint, Initializer
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
    ---
    citation:
        author: Ji
        year: 2015
        link: http://www.aclweb.org/anthology/P15-1067
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        relation_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    #: Secondary embeddings for entities
    entity_projections: Embedding
    #: Secondary embeddings for relations
    relation_projections: Embedding

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        relation_dim: int = 30,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        entity_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        relation_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        **kwargs,
    ) -> None:
        super().__init__(
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=relation_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
                constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
            ),
            **kwargs,
        )

        self.entity_projections = Embedding.init_with_device(
            num_embeddings=self.num_entities,
            embedding_dim=embedding_dim,
            device=self.device,
            initializer=xavier_normal_,
        )
        self.relation_projections = Embedding.init_with_device(
            num_embeddings=self.num_relations,
            embedding_dim=relation_dim,
            device=self.device,
            initializer=xavier_normal_,
        )

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        self.entity_projections.reset_parameters()
        self.relation_projections.reset_parameters()

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
        h_indices: Optional[torch.LongTensor] = None,
        r_indices: Optional[torch.LongTensor] = None,
        t_indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Evaluate the interaction function.

        :param h_indices: shape: (batch_size,)
            The indices of head entities. If None, score against all.
        :param r_indices: shape: (batch_size,)
            The indices of relations. If None, score against all.
        :param t_indices: shape: (batch_size,)
            The indices of tail entities. If None, score against all.

        :return: The scores, shape: (batch_size, num_entities)
        """
        # Head
        h = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        h_p = self.entity_projections.get_in_canonical_shape(indices=h_indices)

        r = self.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        r_p = self.relation_projections.get_in_canonical_shape(indices=r_indices)

        t = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)
        t_p = self.entity_projections.get_in_canonical_shape(indices=t_indices)

        return self.interaction_function(h=h, h_p=h_p, r=r, r_p=r_p, t=t, t_p=t_p)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_indices=hrt_batch[:, 0], r_indices=hrt_batch[:, 1], t_indices=hrt_batch[:, 2])

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_indices=hr_batch[:, 0], r_indices=hr_batch[:, 1], t_indices=None)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_indices=None, r_indices=rt_batch[:, 0], t_indices=rt_batch[:, 1])
