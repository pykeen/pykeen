# -*- coding: utf-8 -*-

"""Implementation of the ComplEx model."""

from typing import Optional

import numpy
import torch
import torch.nn as nn

from ..base import BaseModule
from ..init import embedding_xavier_normal_
from ...customized_loss_functions.softplus_loss import SoftplusLoss
from ...instance_creation_factories import TriplesFactory
from ...typing import OptionalLoss
from ...utils import l2_regularization


class ComplEx(BaseModule):
    """An implementation of ComplEx [trouillon2016]_."""

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 200,
            entity_embeddings: Optional[nn.Embedding] = None,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
            relation_embeddings: Optional[nn.Embedding] = None,
            regularization_factor: float = 0.01,
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
        :param regularization_factor: float
            A weight for the regularization term's contribution relative to the loss value.
        """
        if criterion is None:
            criterion = SoftplusLoss(reduction='mean')

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=2 * embedding_dim,  # complex embeddings
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        # Store the real embedding size
        self.real_embedding_dim = embedding_dim

        # ComplEx uses regularization
        self.regularization_factor = torch.tensor([regularization_factor], requires_grad=False, device=self.device)[0]
        self.current_regularization_term = None

        # The embeddings are first initialized when calling the get_grad_params function
        self.relation_embeddings = relation_embeddings

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        """Initialize entity and relation embeddings."""
        # Initialize entity embeddings
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_normal_(self.entity_embeddings)

        # Initialize relation embeddings
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
            embedding_xavier_normal_(self.relation_embeddings)

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

    def _update_regularization_term(
            self,
            h: torch.FloatTensor,
            r: torch.FloatTensor,
            t: torch.FloatTensor,
    ) -> None:
        """Update the regularization term for the given embeddings.

        :param h: The head embeddings.
        :param r: The relation embeddings.
        :param t: The tail embeddings.
        :return:
        """
        # L2 regularization
        self.current_regularization_term = l2_regularization(h, r, t)

        # normalize by number of elements in tensors
        self.current_regularization_term /= sum(numpy.prod(x.shape) for x in (h, r, t))

    def compute_label_loss(
            self,
            predictions: torch.FloatTensor,
            labels: torch.FloatTensor
    ) -> torch.FloatTensor:  # noqa: D102
        loss = super().compute_label_loss(predictions=predictions, labels=labels)
        loss += self.regularization_factor * self.current_regularization_term
        return loss

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # view as (batch_size, embedding_dim, 2)
        h = self.entity_embeddings(batch[:, 0]).view(-1, self.real_embedding_dim, 2)
        r = self.relation_embeddings(batch[:, 1]).view(-1, self.real_embedding_dim, 2)
        t = self.entity_embeddings(batch[:, 2]).view(-1, self.real_embedding_dim, 2)

        # Update regularization term
        self._update_regularization_term(h=h, r=r, t=t)

        # Compute scores
        scores = self.interaction_function(h=h, r=r, t=t).view(-1, 1)

        return scores

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # view as (batch_size, num_entities, embedding_dim, 2)
        h = self.entity_embeddings(batch[:, 0]).view(-1, 1, self.real_embedding_dim, 2)
        r = self.relation_embeddings(batch[:, 1]).view(-1, 1, self.real_embedding_dim, 2)
        t = self.entity_embeddings.weight.view(1, -1, self.real_embedding_dim, 2)

        # Update regularization term
        self._update_regularization_term(h=h, r=r, t=t)

        # Compute scores
        scores = self.interaction_function(h=h, r=r, t=t)

        return scores

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # view as (batch_size, num_entities, embedding_dim, 2)
        h = self.entity_embeddings.weight.view(1, -1, self.real_embedding_dim, 2)
        r = self.relation_embeddings(batch[:, 0]).view(-1, 1, self.real_embedding_dim, 2)
        t = self.entity_embeddings(batch[:, 1]).view(-1, 1, self.real_embedding_dim, 2)

        # Update regularization term
        self._update_regularization_term(h=h, r=r, t=t)

        # Compute scores
        scores = self.interaction_function(h=h, r=r, t=t)

        return scores
