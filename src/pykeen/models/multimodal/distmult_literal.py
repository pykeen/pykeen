# -*- coding: utf-8 -*-

"""Implementation of the DistMultLiteral model."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from ..base import MultimodalModel
from ...losses import Loss
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint
from ...utils import slice_triples


# TODO: Check entire build of the model
class DistMultLiteral(MultimodalModel):
    """An implementation of DistMultLiteral from [agustinus2018]_."""

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
        input_dropout=dict(type=float, low=0, high=1.0),
    )
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(margin=0.0)

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        input_dropout: float = 0.0,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        numeric_literals = triples_factory.numeric_literals

        # Embeddings
        self.relation_embeddings = None
        self.numeric_literals = nn.Embedding.from_pretrained(
            torch.tensor(numeric_literals, dtype=torch.float, device=self.device), freeze=True,
        )
        # Number of columns corresponds to number of literals
        self.num_of_literals = self.numeric_literals.weight.data.shape[1]
        self.linear_transformation = nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim)
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize the entities and relation embeddings based on the XAVIER initialization."""
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data)
        xavier_normal_(self.relation_embeddings.weight.data)

    @staticmethod
    def _get_embeddings(elements, embedding_module, embedding_dim):
        return embedding_module(elements).view(-1, embedding_dim)

    def _get_literals(self, heads, tails):
        return (
            self._get_embeddings(
                elements=heads,
                embedding_module=self.numeric_literals,
                embedding_dim=self.num_of_literals,
            ),
            self._get_embeddings(
                elements=tails,
                embedding_module=self.numeric_literals,
                embedding_dim=self.num_of_literals,
            ),
        )

    def _get_triple_embeddings(self, heads, relations, tails):
        return (
            self._get_embeddings(
                elements=heads,
                embedding_module=self.entity_embeddings,
                embedding_dim=self.embedding_dim,
            ),
            self._get_embeddings(
                elements=relations,
                embedding_module=self.relation_embeddings,
                embedding_dim=self.embedding_dim,
            ),
            self._get_embeddings(
                elements=tails,
                embedding_module=self.entity_embeddings,
                embedding_dim=self.embedding_dim,
            ),
        )

    def _apply_g_function(self, entity_embeddings, literals):
        """Concatenate the entities with its literals and apply the g function which is a linear transformation in this model.

        :param entity_embeddings: batch_size x self.embedding_dim
        :param literals: batch_size x self.num_literals
        :return:
        """
        return self.linear_transformation(torch.cat([entity_embeddings, literals], dim=1))

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        raise NotImplementedError

    def score_t(self, hr_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass using right side (tail) prediction for training with the LCWA."""
        heads, relations, tails = slice_triples(hr_batch)
        head_embs, relation_embs, tail_embs = self._get_triple_embeddings(
            heads=heads,
            relations=relations,
            tails=tails,
        )
        head_literals, tail_literals = self._get_literals(heads=heads, tails=tails)

        g_heads = self._apply_g_function(entity_embeddings=head_embs, literals=head_literals)
        g_tails = self._apply_g_function(entity_embeddings=tail_embs, literals=tail_literals)

        # apply dropout
        g_heads = self.input_dropout(g_heads)
        g_tails = self.input_dropout(g_tails)

        # -, because lower score shall correspond to a more plausible triple.
        scores = - torch.sum(g_heads * relation_embs * g_tails, dim=1)
        return scores

    # TODO check if this is the same as the BaseModule
    def compute_mr_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        """Compute the mean ranking loss for the positive and negative scores."""
        # Choose y = -1 since a smaller score is better.
        # In TransE for example, the scores represent distances
        if not self.compute_mr_loss:
            raise RuntimeError(
                'The chosen loss does not allow the calculation of Margin Ranking losses. '
                'Please use the compute_label_loss method instead',
            )
        y = torch.ones_like(negative_scores, device=self.device) * -1
        loss = self.loss(positive_scores, negative_scores, y)
        return loss
