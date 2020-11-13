# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Optional

import torch
from torch import nn

from ..base import InteractionFunction, SimpleVectorEntityRelationEmbeddingModel
from ...losses import Loss
from ...nn import functional as pykeen_functional
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

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
        self.hidden = nn.Linear(in_features=3 * embedding_dim, out_features=hidden_dim, bias=True)
        self.activation = nn.ReLU()
        self.hidden_to_score = nn.Linear(in_features=hidden_dim, out_features=1, bias=True)

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_for_empty_kwargs(kwargs)
        return pykeen_functional.ermlp_interaction(
            h=h,
            r=r,
            t=t,
            hidden=self.hidden,
            activation=self.activation,
            final=self.hidden_to_score,
        )

    def reset_parameters(self):  # noqa: D102
        # Initialize biases with zero
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.hidden_to_score.bias)
        # In the original formulation,
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(
            self.hidden_to_score.weight,
            gain=nn.init.calculate_gain(self.activation.__class__.__name__.lower()),
        )


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
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model."""
        interaction_function = ERMLPInteractionFunction(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

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
