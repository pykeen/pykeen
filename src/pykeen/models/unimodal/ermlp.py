# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory

__all__ = [
    'ERMLP',
]


class ERMLP(EntityRelationEmbeddingModel):
    """An implementation of ERMLP from [dong2014]_.

    This model uses a neural network-based approach.
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        if hidden_dim is None:
            hidden_dim = embedding_dim
        self.hidden_dim = hidden_dim
        """The multi-layer perceptron consisting of an input layer with 3 * self.embedding_dim neurons, a  hidden layer
           with self.embedding_dim neurons and output layer with one neuron.
           The input is represented by the concatenation embeddings of the heads, relations and tail embeddings.
        """
        self.linear1 = nn.Linear(3 * self.embedding_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 1)
        self.mlp = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
        )

        # Finalize initialization
        self.reset_parameters_()

    def _reset_parameters_(self):  # noqa: D102
        # The authors do not specify which initialization was used. Hence, we use the pytorch default.
        self.entity_embeddings.reset_parameters()
        self.relation_embeddings.reset_parameters()

        # weight initialization
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain('relu'))

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hrt_batch[:, 0])
        r = self.relation_embeddings(hrt_batch[:, 1])
        t = self.entity_embeddings(hrt_batch[:, 2])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        # Concatenate them
        x_s = torch.cat([h, r, t], dim=-1)

        # Compute scores
        return self.mlp(x_s)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hr_batch[:, 0])
        r = self.relation_embeddings(hr_batch[:, 1])
        t = self.entity_embeddings.weight

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        # First layer can be unrolled
        layers = self.mlp.children()
        first_layer = next(layers)
        w = first_layer.weight
        i = 2 * self.embedding_dim
        w_hr = w[None, :, :i] @ torch.cat([h, r], dim=-1).unsqueeze(-1)
        w_t = w[None, :, i:] @ t.unsqueeze(-1)
        b = first_layer.bias
        scores = (b[None, None, :] + w_hr[:, None, :, 0]) + w_t[None, :, :, 0]

        # Send scores through rest of the network
        scores = scores.view(-1, self.hidden_dim)
        for remaining_layer in layers:
            scores = remaining_layer(scores)
        scores = scores.view(-1, self.num_entities)
        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(rt_batch[:, 0])
        t = self.entity_embeddings(rt_batch[:, 1])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        # First layer can be unrolled
        layers = self.mlp.children()
        first_layer = next(layers)
        w = first_layer.weight
        i = self.embedding_dim
        w_h = w[None, :, :i] @ h.unsqueeze(-1)
        w_rt = w[None, :, i:] @ torch.cat([r, t], dim=-1).unsqueeze(-1)
        b = first_layer.bias
        scores = (b[None, None, :] + w_rt[:, None, :, 0]) + w_h[None, :, :, 0]

        # Send scores through rest of the network
        scores = scores.view(-1, self.hidden_dim)
        for remaining_layer in layers:
            scores = remaining_layer(scores)
        scores = scores.view(-1, self.num_entities)

        return scores
