# -*- coding: utf-8 -*-

"""An implementation of the extension to ERMLP."""

from typing import Any, ClassVar, Mapping, Type

import torch
from torch import nn
from torch.nn.init import uniform_

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn.emb import EmbeddingSpecification
from ...typing import Hint, Initializer

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
    ---
    citation:
        author: Sharifzadeh
        year: 2019
        link: https://github.com/pykeen/pykeen
        github: pykeen/pykeen
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        hidden_dim=dict(type=int, low=5, high=9, scale='power_two'),
        input_dropout=DEFAULT_DROPOUT_HPO_RANGE,
        hidden_dropout=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        *,
        hidden_dim: int = 300,
        input_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        embedding_dim: int = 200,
        entity_initializer: Hint[Initializer] = uniform_,
        relation_initializer: Hint[Initializer] = uniform_,
        **kwargs,
    ) -> None:
        super().__init__(
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
        )
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(2 * self.embedding_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout)
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
