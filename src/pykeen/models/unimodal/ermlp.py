# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Optional

import torch.autograd
from torch import nn

from ..base import EntityRelationEmbeddingModel, InteractionFunction
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...utils import get_embedding_in_canonical_shape

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
        self.interaction_function = ERMLPInteractionFunction(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        # Finalize initialization
        self.reset_parameters_()

    def _reset_parameters_(self):  # noqa: D102
        # The authors do not specify which initialization was used. Hence, we use the pytorch default.
        self.entity_embeddings.reset_parameters()
        self.relation_embeddings.reset_parameters()
        self.interaction_function.reset_parameters()

    def _score(
        self,
        h_ind: Optional[torch.LongTensor] = None,
        r_ind: Optional[torch.LongTensor] = None,
        t_ind: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # Get embeddings
        h = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=h_ind)
        r = get_embedding_in_canonical_shape(embedding=self.relation_embeddings, ind=r_ind)
        t = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=t_ind)

        # Compute score
        scores = self.interaction_function(h=h, r=r, t=t)

        # Only regularize relation embeddings
        self.regularize_if_necessary(r)

        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hrt_batch[:, 0], r_ind=hrt_batch[:, 1], t_ind=hrt_batch[:, 2]).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hr_batch[:, 0], r_ind=hr_batch[:, 1], t_ind=None).view(-1, self.num_entities)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=None, r_ind=rt_batch[:, 0], t_ind=rt_batch[:, 1]).view(-1, self.num_entities)


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
