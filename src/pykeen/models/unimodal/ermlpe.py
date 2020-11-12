# -*- coding: utf-8 -*-

"""An implementation of the extension to ERMLP."""

from typing import Optional, Type

import torch
from torch import nn

from ..base import EntityRelationEmbeddingModel, InteractionFunction
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn import functional as pykeen_functional
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ERMLPE',
]


class ERMLPEInteractionFunction(InteractionFunction):
    """Interaction function of ER-MLP."""

    def __init__(
        self,
        hidden_dim: int = 300,
        input_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        embedding_dim: int = 200,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.Dropout(hidden_dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(hidden_dropout),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_for_empty_kwargs(kwargs=kwargs)
        return pykeen_functional.ermlpe_interaction(h=h, r=r, t=t, mlp=self.mlp)


class ERMLPE(EntityRelationEmbeddingModel):
    r"""An extension of ERMLP proposed by [sharifzadeh2019]_.

    This model uses a neural network-based approach similar to ERMLP and with slight modifications.
    In ERMLP, the model is:

    .. math::

        f(h, r, t) = \textbf{w}^{T} g(\textbf{W} [\textbf{h}; \textbf{r}; \textbf{t}])

    whereas in ERMPLE the model is:

    .. math::

        f(h, r, t) = \textbf{t}^{T} f(\textbf{W} (g(\textbf{W} [\textbf{h}; \textbf{r}]))

    including dropouts and batch-norms between each two hidden layers.
    ConvE can be seen as a special case of ERMLPE that contains the unnecessary inductive bias of convolutional
    filters. The aim of this model is to show that lifting this bias from ConvE (which simply leaves us with a
    modified ERMLP model), not only reduces the number of parameters but also improves performance.

    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
        hidden_dim=dict(type=int, low=50, high=450, q=25),
        input_dropout=dict(type=float, low=0.0, high=0.8, q=0.1),
        hidden_dropout=dict(type=float, low=0.0, high=0.8, q=0.1),
    )
    #: The default loss function class
    loss_default: Type[Loss] = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = {}

    def __init__(
        self,
        triples_factory: TriplesFactory,
        hidden_dim: int = 300,
        input_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        embedding_dim: int = 200,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.interaction_function = ERMLPEInteractionFunction(
            hidden_dim=hidden_dim,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout,
            embedding_dim=embedding_dim,
        )

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        self.interaction_function.reset_parameters()

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0])
        r = self.relation_embeddings(indices=hrt_batch[:, 1])
        t = self.entity_embeddings(indices=hrt_batch[:, 2])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        return self.interaction_function.score_hrt(h=h, r=r, t=t)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=hr_batch[:, 0])
        r = self.relation_embeddings(indices=hr_batch[:, 1])
        t = self.entity_embeddings(indices=None)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        return self.interaction_function.score_t(h=h, r=r, all_entities=t)

    def score_r(self, ht_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=ht_batch[:, 0])
        r = self.relation_embeddings(indices=None)
        t = self.entity_embeddings(indices=ht_batch[:, 1])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        return self.interaction_function.score_r(h=h, all_relations=r, t=t)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=None)
        r = self.relation_embeddings(indices=rt_batch[:, 0])
        t = self.entity_embeddings(indices=rt_batch[:, 1])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        return self.interaction_function.score_h(all_entities=h, r=r, t=t)
