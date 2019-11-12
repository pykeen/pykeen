# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ...losses import Loss
from ...triples import TriplesFactory

__all__ = [
    'ERMLP',
]


class ERMLP(BaseModule):
    """An implementation of ERMLP from [dong2014]_.

    This model uses a neural network-based approach.
    """

    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        entity_embeddings: Optional[nn.Embedding] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init: bool = True,
    ) -> None:
        """Initialize the model."""
        if hidden_dim is None:
            hidden_dim = embedding_dim

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

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

        self.relation_embeddings = relation_embeddings

        if init:
            self.init_empty_weights_()

    def init_empty_weights_(self):  # noqa: D102
        # The authors do not specify which initialization was used. Hence, we use the pytorch default.
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        # TODO: How to determine whether weights have been initialized
        # weight (re-)initialization
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain('relu'))

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.relation_embeddings = None
        return self

    def forward_owa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings(batch[:, 2])

        # Concatenate them
        x_s = torch.cat([h, r, t], dim=-1)

        # Compute scores
        return self.mlp(x_s)

    def forward_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(batch[:, 0])
        r = self.relation_embeddings(batch[:, 1])
        t = self.entity_embeddings.weight

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

    def forward_inverse_cwa(self, batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(batch[:, 0])
        t = self.entity_embeddings(batch[:, 1])

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
