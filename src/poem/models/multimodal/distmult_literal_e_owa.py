# -*- coding: utf-8 -*-

"""Implementation of the DistMultLiteral model."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from ..base import BaseModule, slice_triples
from ...constants import DISTMULT_LITERAL_NAME_OWA, INPUT_DROPOUT, NUMERIC_LITERALS

# TODO: Check entire build of the model
class DistMultLiteral(BaseModule):
    """
    An implementation of DistMultLiteral [agustinus2018] based on the open world assumption (OWA)

    .. [agustinus2018] Kristiadi, Agustinus, et al. "Incorporating literals into knowledge graph embeddings."
                       arXiv preprint arXiv:1802.00934 (2018).
    """
    model_name = DISTMULT_LITERAL_NAME_OWA
    margin_ranking_loss_average: bool = True

    def __init__(self) -> None:
        super().__init__()

        # TODO: Check this
        # numeric_literals = model_config.multimodal_data.get(NUMERIC_LITERALS)

        # Embeddings
        self.relation_embeddings = None
        # self.numeric_literals = nn.Embedding.from_pretrained(
        #     torch.tensor(numeric_literals, dtype=torch.float, device=self.device), freeze=True)
        # Number of columns corresponds to number of literals
        self.num_of_literals = self.numeric_literals.weight.data.shape[1]
        self.linear_transformation = nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim)
        self.input_dropout = torch.nn.Dropout(self.config[INPUT_DROPOUT]
                                              if INPUT_DROPOUT in self.config else 0.)


    def _init_embeddings(self):
        """Initialize the entities and relation embeddings based on the XAVIER initialization."""
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data)
        xavier_normal_(self.relation_embeddings.weight.data)

    def _get_literals(self, heads, tails):
        """"""
        return (self._get_embeddings(elements=heads,
                                     embedding_module=self.numeric_literals,
                                     embedding_dim=self.num_of_literals),
                self._get_embeddings(elements=tails,
                                     embedding_module=self.numeric_literals,
                                     embedding_dim=self.num_of_literals),
                )

    def _get_triple_embeddings(self, heads, relations, tails):
        """"""
        return (self._get_embeddings(elements=heads,
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

    def forward(self, triples):
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

        # -, because lower score shall correspond to a more plausible triple.
        scores = - torch.sum(g_heads * relation_embs * g_tails, dim=1)
        return scores

    def compute_mr_loss(self, pos_triple_scores: torch.Tensor, neg_triples_scores: torch.Tensor) -> torch.Tensor:
        """"""
        # Choose y = -1 since a smaller score is better.
        # In TransE for example, the scores represent distances
        assert self.compute_mr_loss == True,\
            'The chosen criterion does not allow the calculation of Margin Ranking losses. Please use the' \
            'compute_label_loss method instead'
        y = torch.ones_like(neg_triples_scores, device=self.device) * -1
        loss = self.criterion(pos_triple_scores, neg_triples_scores, y)
        return loss
