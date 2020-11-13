# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from typing import Optional

import torch
import torch.autograd

from .. import Model
from ..base import GeneralVectorEntityRelationEmbeddingModel, IndexFunction
from ...losses import Loss
from ...nn import Embedding
from ...nn.init import xavier_normal_
from ...nn.modules import InteractionFunction, TranslationalInteractionFunction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import clamp_norm

__all__ = [
    'TransDInteractionFunction',
    'TransDIndexFunction',
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


class TransDInteractionFunction(TranslationalInteractionFunction):
    """The interaction function for TransD."""

    def __init__(self, p: int = 2, power: int = 2):
        """Initialize the TransD interaction function.

        :param p: The norm applied by :func:`torch.norm`
        :param power: The power applied after :func:`torch.norm`.
        """
        super().__init__(p=p)
        self.power = power

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa:D102
        return super().forward(h=h, r=r, t=t, **kwargs) ** self.power


class TransDIndexFunction(IndexFunction):
    """The index-based interaction function for TransD."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        relation_dim: int,
        device: DeviceHint,
        interaction_function: Optional[InteractionFunction] = None,
    ):
        super().__init__()
        self.entity_projections = Embedding.init_with_device(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim,
            device=device,
            initializer=xavier_normal_,
        )
        self.relation_projections = Embedding.init_with_device(
            num_embeddings=num_relations,
            embedding_dim=relation_dim,
            device=device,
            initializer=xavier_normal_,
        )
        if interaction_function is None:
            interaction_function = TransDInteractionFunction()
        self.interaction_function = interaction_function

    def reset_parameters(self):  # noqa: D102
        self.entity_projections.reset_parameters()
        self.relation_projections.reset_parameters()
        self.interaction_function.reset_parameters()

    def forward(
        self,
        model: Model,
        h_indices: Optional[torch.LongTensor] = None,
        r_indices: Optional[torch.LongTensor] = None,
        t_indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        h = model.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        h_p = self.entity_projections.get_in_canonical_shape(indices=h_indices)

        r = model.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        r_p = self.relation_projections.get_in_canonical_shape(indices=r_indices)

        t = model.entity_embeddings.get_in_canonical_shape(indices=t_indices)
        t_p = self.entity_projections.get_in_canonical_shape(indices=t_indices)

        # Project entities
        h_bot = _project_entity(e=h, e_p=h_p, r=r, r_p=r_p)
        t_bot = _project_entity(e=t, e_p=t_p, r=r, r_p=r_p)

        return self.interaction_function(h=h_bot, r=r, t=t_bot)


class TransD(GeneralVectorEntityRelationEmbeddingModel):
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
        index_function = TransDIndexFunction(
            num_entities=triples_factory.num_entities,
            num_relations=triples_factory.num_relations,
            embedding_dim=embedding_dim,
            relation_dim=relation_dim,
            device=preferred_device,
        )

        super().__init__(
            triples_factory=triples_factory,
            index_function=index_function,
            embedding_dim=embedding_dim,
            relation_dim=relation_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            entity_initializer=xavier_normal_,
            relation_initializer=xavier_normal_,
            entity_constrainer=clamp_norm,
            entity_constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
            relation_constrainer=clamp_norm,
            relation_constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
        )
