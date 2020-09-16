# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Optional

import torch.autograd

from ..base import ERMLPInteractionFunction, EntityRelationEmbeddingModel
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
