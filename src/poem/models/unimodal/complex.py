# -*- coding: utf-8 -*-

"""Implementation of the ComplEx model."""

from typing import Optional

import torch
import torch.nn as nn

from ..base import BaseModule
from ..init import embedding_xavier_normal_
from ...losses import Loss, SoftplusLoss
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...utils import resolve_device

__all__ = [
    'ComplEx',
]


class ComplEx(BaseModule):
    """An implementation of ComplEx [trouillon2016]_."""

    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        regularization_weight=dict(type=float, low=0.0, high=0.1, scale='log'),
    )

    criterion_default = SoftplusLoss
    criterion_default_kwargs = dict(reduction='mean')

    #: The regularizer used by [trouillon2016]_ for ComplEx.
    regularizer_default = LpRegularizer
    #: The LP settings used by [trouillon2016]_ for ComplEx.
    regularizer_default_kwargs = dict(
        weight=0.01,
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 200,
        entity_embeddings: Optional[nn.Embedding] = None,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the module.

        :param triples_factory: TriplesFactory
            The triple factory connected to the model.
        :param embedding_dim: int
            The embedding dimensionality of the entity embeddings.
        :param entity_embeddings: nn.Embedding (optional)
            Initialization for the entity embeddings.
        :param criterion: OptionalLoss (optional)
            The loss criterion to use. Defaults to SoftplusLoss.
        :param preferred_device: str (optional)
            The default device where to model is located.
        :param random_seed: int (optional)
            An optional random seed to set before the initialization of weights.
        :param relation_embeddings: nn.Embedding (optional)
            Relation embeddings initialization.
        :param regularizer: BaseRegularizer
            The regularizer to use.
        """
        if regularizer == 'troullion2016':
            # In the paper, they use weight of 0.01, and normalize the regularization term by the number of elements
            regularizer = LpRegularizer(
                device=resolve_device(preferred_device),
                weight=0.01,
                p=2.0,
                normalize=True,
            )

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=2 * embedding_dim,  # complex embeddings
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        # Store the real embedding size
        self.real_embedding_dim = embedding_dim

        # The embeddings are first initialized when calling the get_grad_params function
        self.relation_embeddings = relation_embeddings

        # Finalize initialization
        self._init_weights_on_device()

    def init_empty_weights_(self):  # noqa: D102
        # Initialize entity embeddings
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_normal_(self.entity_embeddings)

        # Initialize relation embeddings
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
            embedding_xavier_normal_(self.relation_embeddings)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.relation_embeddings = None
        return self

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function of ComplEx for given embeddings.

        The embeddings have to be in a broadcastable shape.

        :param h: shape: (..., e, 2)
            Head embeddings. Last dimension corresponds to (real, imag).
        :param r: shape: (..., e, 2)
            Relation embeddings. Last dimension corresponds to (real, imag).
        :param t: shape: (..., e, 2)
            Tail embeddings. Last dimension corresponds to (real, imag).

        :return: shape: (...)
            The scores.
        """
        assert all(x.shape[-1] == 2 for x in (h, r, t))

        # ComplEx space bilinear product (equivalent to HolE)
        # *: Elementwise multiplication
        re_re_re = h[..., 0] * r[..., 0] * t[..., 0]
        re_im_im = h[..., 0] * r[..., 1] * t[..., 1]
        im_re_im = h[..., 1] * r[..., 0] * t[..., 1]
        im_im_re = h[..., 1] * r[..., 1] * t[..., 0]
        scores = torch.sum(re_re_re + re_im_im + im_re_im - im_im_re, dim=-1)

        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # view as (batch_size, embedding_dim, 2)
        h = self.entity_embeddings(hrt_batch[:, 0])
        r = self.relation_embeddings(hrt_batch[:, 1])
        t = self.entity_embeddings(hrt_batch[:, 2])
        reg_shape = (-1, self.real_embedding_dim, 2)

        # Compute scores
        scores = self.interaction_function(h=h.view(reg_shape), r=r.view(reg_shape), t=t.view(reg_shape)).view(-1, 1)

        # Regularization
        reg_shape = (-1, self.embedding_dim)
        self.regularize_if_necessary(h.view(reg_shape), r.view(reg_shape), t.view(reg_shape))

        return scores

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # view as (hr_batch_size, num_entities, embedding_dim, 2)
        h = self.entity_embeddings(hr_batch[:, 0])
        r = self.relation_embeddings(hr_batch[:, 1])
        t = self.entity_embeddings.weight
        reg_shape = (-1, 1, self.real_embedding_dim, 2)
        new_shape_tails = (1, -1, self.real_embedding_dim, 2)

        # Compute scores
        scores = self.interaction_function(h=h.view(reg_shape), r=r.view(reg_shape), t=t.view(new_shape_tails))

        # Regularization
        reg_shape = (-1, 1, self.embedding_dim)
        new_shape_tails = (1, -1, self.embedding_dim)
        self.regularize_if_necessary(h.view(reg_shape), r.view(reg_shape), t.view(new_shape_tails))

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # view as (rt_batch_size, num_entities, embedding_dim, 2)
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(rt_batch[:, 0])
        t = self.entity_embeddings(rt_batch[:, 1])
        reg_shape = (-1, 1, self.real_embedding_dim, 2)
        new_shape_heads = (1, -1, self.real_embedding_dim, 2)

        # Compute scores
        scores = self.interaction_function(h=h.view(new_shape_heads), r=r.view(reg_shape), t=t.view(reg_shape))

        # Regularization
        reg_shape = (-1, 1, self.embedding_dim)
        new_shape_heads = (1, -1, self.embedding_dim)
        self.regularize_if_necessary(h.view(new_shape_heads), r.view(reg_shape), t.view(reg_shape))

        return scores
