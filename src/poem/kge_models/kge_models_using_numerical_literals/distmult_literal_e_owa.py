# -*- coding: utf-8 -*-

"""Implementation of the DistMultLiteral model."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from poem.constants import DISTMULT_LITERAL_NAME_OWA, DISTMULT_INPUT_DROPOUT, NUMERIC_LITERALS
from poem.kge_models.base_owa import BaseOWAModule, slice_triples
from poem.model_config import ModelConfig


class DistMultLiteral(BaseOWAModule):
    """
    An implementation of DistMultLiteral [agustinus2018] based on the open world assumption (OWA)

    .. [agustinus2018] Kristiadi, Agustinus, et al. "Incorporating literals into knowledge graph embeddings."
                       arXiv preprint arXiv:1802.00934 (2018).
    """
    model_name = DISTMULT_LITERAL_NAME_OWA
    margin_ranking_loss_average: bool = True

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)

        numeric_literals = model_config.multimodal_data.get(NUMERIC_LITERALS)

        # Embeddings
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.numeric_literals = nn.Embedding.from_pretrained(
            torch.tensor(numeric_literals, dtype=torch.float, device=self.device), freeze=True)
        # Number of columns corresponds to number of literals
        self.num_of_literals = self.numeric_literals.weight.data.shape[1]
        self.linear_transformation = nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim)
        self.input_dropout = torch.nn.Dropout(
            self.config[DISTMULT_INPUT_DROPOUT] if DISTMULT_INPUT_DROPOUT in self.config else 0.)

        self._initialize()

    def _initialize(self):
        """Initialize the entities and relation embeddings based on the XAVIER initialization."""
        xavier_normal_(self.entity_embeddings.weight.data)
        xavier_normal_(self.relation_embeddings.weight.data)

    def _get_literals(self, heads, tails):
        """"""
        return (
            self._get_embeddings(elements=heads,
                                 embedding_module=self.numeric_literals,
                                 embedding_dim=self.num_of_literals),
            self._get_embeddings(elements=tails,
                                 embedding_module=self.numeric_literals,
                                 embedding_dim=self.num_of_literals),
        )

    def _get_triple_embeddings(self, heads, relations, tails):
        """"""
        return (
            self._get_embeddings(elements=heads,
                                 embedding_module=self.entity_embeddings,
                                 embedding_dim=self.embedding_dim),
            self._get_embeddings(elements=relations,
                                 embedding_module=self.relation_embeddings,
                                 embedding_dim=self.embedding_dim),
            self._get_embeddings(elements=tails,
                                 embedding_module=self.entity_embeddings,
                                 embedding_dim=self.embedding_dim),
        )

    def _apply_g_function(self, entity_embeddings, literals):
        """
        Concatenate the entities with its literals and apply the g function which is a linear transformation
        in this model.
        :param entity_embeddings: batch_size x self.embedding_dim
        :param literals: batch_size x self.num_literals
        :return:
        """
        return self.linear_transformation(torch.cat([entity_embeddings, literals], dim=1))

    def _compute_scores(self, head_embs, relation_embs, tail_embs):
        """
        Compute scores based on DistMult's scoring function.
        :param head_embs: batch_size x self.embedding_dim
        :param relation_embs: batch_size x self.embedding_dim
        :param tail_embs: batch_size x self.embedding_dim
        :return:
        """

        # -, because lower score shall correspond to a more plausible triple.
        scores = - torch.sum(head_embs * relation_embs * tail_embs, dim=1)
        return scores

    def _score_triples(self, triples):
        """"""
        heads, relations, tails = slice_triples(triples)
        head_embs, relation_embs, tail_embs = self._get_triple_embeddings(heads=heads,
                                                                          relations=relations,
                                                                          tails=tails)
        head_literals, tail_literals = self._get_literals(heads=heads, tails=tails)

        g_heads = self._apply_g_function(entity_embeddings=head_embs, literals=head_literals)
        g_tails = self._apply_g_function(entity_embeddings=tail_embs, literals=tail_literals)

        # apply dropout
        g_heads = self.input_dropout(g_heads)
        g_tails = self.input_dropout(g_tails)

        return self._compute_scores(head_embs=g_heads, relation_embs=relation_embs, tail_embs=g_tails)

    def _compute_loss(self, positive_scores, negative_scores):
        # Choose y = -1 since a smaller score is better.
        # In TransE for example, the scores represent distances
        y = np.repeat([-1], repeats=positive_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        loss = self.criterion(positive_scores, negative_scores, y)
        return loss

    def predict(self, triples: torch.tensor) -> np.array:
        """
        Compute predictions.
        :param triples: num_triples x 3
        :return:
        """
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        """
        Performs all the computation by the model for the passed batch.
        :param batch_positives: batch_size x 3
        :param batch_negatives: batch_size x 3
        :return:
        """

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss
