# -*- coding: utf-8 -*-

"""An implementation of the extension to ERMLP."""

from typing import Optional, Type

import torch
from torch import nn

from ..base import EntityRelationEmbeddingModel
from ...losses import BCEAfterSigmoidLoss, Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ERMLPE',
]


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
        self.hidden_dim = hidden_dim
        self.input_dropout = input_dropout

        self.linear1 = nn.Linear(2 * self.embedding_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.input_dropout = nn.Dropout(self.input_dropout)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim)
        self.mlp = nn.Sequential(
            self.linear1,
            nn.Dropout(hidden_dropout),
            self.bn1,
            nn.ReLU(),
            self.linear2,
            nn.Dropout(hidden_dropout),
            self.bn2,
            nn.ReLU(),
        )

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        for module in [
            self.linear1,
            self.linear2,
            self.bn1,
            self.bn2,
        ]:
            module.reset_parameters()

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0]).view(-1, self.embedding_dim)
        r = self.relation_embeddings(indices=hrt_batch[:, 1]).view(-1, self.embedding_dim)
        t = self.entity_embeddings(indices=hrt_batch[:, 2])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        # Concatenate them
        x_s = torch.cat([h, r], dim=-1)
        x_s = self.input_dropout(x_s)

        # Predict t embedding
        x_t = self.mlp(x_s)

        # compare with all t's
        # For efficient calculation, each of the calculated [h, r] rows has only to be multiplied with one t row
        x = (x_t.view(-1, self.embedding_dim) * t).sum(dim=1, keepdim=True)
        # The application of the sigmoid during training is automatically handled by the default loss.

        return x

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=hr_batch[:, 0]).view(-1, self.embedding_dim)
        r = self.relation_embeddings(indices=hr_batch[:, 1]).view(-1, self.embedding_dim)
        t = self.entity_embeddings(indices=None).transpose(1, 0)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        # Concatenate them
        x_s = torch.cat([h, r], dim=-1)
        x_s = self.input_dropout(x_s)

        # Predict t embedding
        x_t = self.mlp(x_s)

        x = x_t @ t
        # The application of the sigmoid during training is automatically handled by the default loss.

        return x

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(indices=None)
        r = self.relation_embeddings(indices=rt_batch[:, 0]).view(-1, self.embedding_dim)
        t = self.entity_embeddings(indices=rt_batch[:, 1]).view(-1, self.embedding_dim)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        rt_batch_size = t.shape[0]

        # Extend each rt_batch of "r" with shape [rt_batch_size, dim] to [rt_batch_size, dim * num_entities]
        r = torch.repeat_interleave(r, self.num_entities, dim=0)
        # Extend each h with shape [num_entities, dim] to [rt_batch_size * num_entities, dim]
        # h = torch.repeat_interleave(h, rt_batch_size, dim=0)
        h = h.repeat(rt_batch_size, 1)

        # Extend t
        t = t.repeat_interleave(self.num_entities, dim=0)

        # Concatenate them
        x_s = torch.cat([h, r], dim=-1)
        x_s = self.input_dropout(x_s)

        # Predict t embedding
        x_t = self.mlp(x_s)

        # For efficient calculation, each of the calculated [h, r] rows has only to be multiplied with one t row
        x = (x_t.view(-1, self.embedding_dim) * t).sum(dim=1, keepdim=True)
        # The results have to be realigned with the expected output of the score_h function
        x = x.view(rt_batch_size, self.num_entities)
        # The application of the sigmoid during training is automatically handled by the default loss.

        return x
