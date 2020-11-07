# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

import math
from typing import Optional

import torch
from torch import nn

from ..base import InteractionFunction, SimpleVectorEntityRelationEmbeddingModel
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory

__all__ = [
    'ERMLP',
    'ERMLPInteractionFunction',
]


class ERMLPInteractionFunction(InteractionFunction):
    """
    Interaction function of ER-MLP.

    .. math ::
        f(h, r, t) = W_2 ReLU(W_1 cat(h, r, t) + b_1) + b_2
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ):
        """
        Initialize the interaction function.

        :param embedding_dim:
            The embedding vector dimension.
        :param hidden_dim:
            The hidden dimension of the MLP.
        """
        super().__init__()
        """The multi-layer perceptron consisting of an input layer with 3 * self.embedding_dim neurons, a  hidden layer
           with self.embedding_dim neurons and output layer with one neuron.
           The input is represented by the concatenation embeddings of the heads, relations and tail embeddings.
        """
        self.head_to_hidden = nn.Linear(in_features=embedding_dim, out_features=hidden_dim, bias=False)
        self.rel_to_hidden = nn.Linear(in_features=embedding_dim, out_features=hidden_dim, bias=True)
        self.tail_to_hidden = nn.Linear(in_features=embedding_dim, out_features=hidden_dim, bias=False)
        self.activation = nn.ReLU()
        self.hidden_to_score = nn.Linear(in_features=hidden_dim, out_features=1, bias=True)

    @classmethod
    def from_model(cls, module: 'ERMLP'):  # noqa:D102
        return cls(embedding_dim=module.embedding_dim, hidden_dim=module.hidden_dim)

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        h = self.head_to_hidden(h)
        r = self.rel_to_hidden(r)
        t = self.tail_to_hidden(t)
        # TODO: Choosing which to combine first, h/r, h/t or r/t, depending on the shape might further improve
        #       performance in a 1:n scenario.
        x = self.activation(h[:, :, None, None, :] + r[:, None, :, None, :] + t[:, None, None, :, :])
        return self.hidden_to_score(x).squeeze(dim=-1)

    def reset_parameters(self):  # noqa: D102
        # Initialize biases with zero
        nn.init.zeros_(self.rel_to_hidden.bias)
        nn.init.zeros_(self.hidden_to_score.bias)
        # In the original formulation,
        #   W_2 sigma(W_1 cat([h, r, t]) + b_1) + b_2
        # W_1 would be initialized with nn.init.xavier_uniform, i.e. with a samples from uniform(-a, a) with
        # a = math.sqrt(3.0) * gain * math.sqrt(2.0 / float(fan_in + fan_out))
        # we have:
        # fan_out = hidden_dim
        # fan_in = 3 * embedding_dim
        bound = math.sqrt(3.0) * 1 * math.sqrt(2.0 / float(sum(self.head_to_hidden.weight.shape)))
        for mod in [
            self.head_to_hidden,
            self.rel_to_hidden,
            self.tail_to_hidden,
        ]:
            nn.init.uniform_(mod.weight, -bound, bound)
        nn.init.xavier_uniform_(self.hidden_to_score.weight, gain=nn.init.calculate_gain('relu'))


class ERMLP(SimpleVectorEntityRelationEmbeddingModel):
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
        if hidden_dim is None:
            self.hidden_dim = embedding_dim

        interaction_function = ERMLPInteractionFunction.from_model(self)

        super().__init__(
            triples_factory=triples_factory,
            interaction_function=interaction_function,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
