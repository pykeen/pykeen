# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Optional

from ..base import ERModel
from ...losses import Loss
from ...nn import EmbeddingSpecification
from ...nn.modules import ERMLPInteraction
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ERMLP',
]


class ERMLP(ERModel):
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
    ) -> None:
        """Initialize ERMLP."""
        if hidden_dim is None:
            hidden_dim = embedding_dim

        super().__init__(
            triples_factory=triples_factory,
            interaction=ERMLPInteraction(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            ),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
            ),
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
