# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Any, ClassVar, Mapping, Optional

import torch
import torch.autograd
from torch import nn
from torch.nn.init import uniform_

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss
from ...nn import EmbeddingSpecification
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint, Hint, Initializer

__all__ = [
    'ERMLP',
]


class ERMLP(EntityRelationEmbeddingModel):
    r"""An implementation of ERMLP from [dong2014]_.

    ERMLP is a multi-layer perceptron based approach that uses a single hidden layer and represents entities and
    relations as vectors. In the input-layer, for each triple the embeddings of head, relation, and tail are
    concatenated and passed to the hidden layer. The output-layer consists of a single neuron that computes the
    plausibility score of the triple:

    .. math::

        f(h,r,t) = \textbf{w}^{T} g(\textbf{W} [\textbf{h}; \textbf{r}; \textbf{t}]),

    where $\textbf{W} \in \mathbb{R}^{k \times 3d}$ represents the weight matrix of the hidden layer,
    $\textbf{w} \in \mathbb{R}^{k}$, the weights of the output layer, and $g$ denotes an activation function such
    as the hyperbolic tangent.
    ---
    citation:
        author: Dong
        year: 2014
        link: https://dl.acm.org/citation.cfm?id=2623623
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
        entity_initializer: Hint[Initializer] = uniform_,
        relation_initializer: Hint[Initializer] = uniform_,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=relation_initializer,
            ),
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

    def _reset_parameters_(self):  # noqa: D102
        # The authors do not specify which initialization was used. Hence, we use the pytorch default.
        super()._reset_parameters_()

        # weight initialization
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain('relu'))

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0])
        r = self.relation_embeddings(indices=hrt_batch[:, 1])
        t = self.entity_embeddings(indices=hrt_batch[:, 2])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        # Concatenate them
        x_s = torch.cat([h, r, t], dim=-1)

        # Compute scores
        return self.mlp(x_s)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0])
        r = self.relation_embeddings(indices=hr_batch[:, 1])
        t = self.entity_embeddings(indices=None)

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
        h = self.entity_embeddings(indices=None)
        r = self.relation_embeddings(indices=rt_batch[:, 0])
        t = self.entity_embeddings(indices=rt_batch[:, 1])

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
