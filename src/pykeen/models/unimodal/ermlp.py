# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Any, ClassVar, Mapping, Optional

import torch
from torch import nn
from torch.nn.init import uniform_

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...typing import Hint, Initializer

__all__ = [
    "ERMLP",
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
    name: ER-MLP
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
        *,
        embedding_dim: int = 50,
        hidden_dim: Optional[int] = None,
        entity_initializer: Hint[Initializer] = uniform_,
        relation_initializer: Hint[Initializer] = uniform_,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
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

    # docstr-coverage: inherited
    def _reset_parameters_(self):  # noqa: D102
        # The authors do not specify which initialization was used. Hence, we use the pytorch default.
        super()._reset_parameters_()

        # weight initialization
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain("relu"))

    # docstr-coverage: inherited
    def score_hrt(self, hrt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
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

    # docstr-coverage: inherited
    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
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

    # docstr-coverage: inherited
    def score_h(self, rt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
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
