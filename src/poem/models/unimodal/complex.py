# -*- coding: utf-8 -*-

"""Implementation of the Complex model based on the closed world assumption (CWA)."""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from poem.constants import CWA, GPU, COMPLEX_NAME
from poem.models.base_owa import BaseOWAModule, slice_triples


class ComplEx(BaseOWAModule):
    """An implementation of ComplEx [Trouillon2016complex].

    .. [trouillon2016complex] Trouillon, Théo, et al. "Complex embeddings for simple link prediction."
                              International Conference on Machine Learning. 2016.
    """
    model_name = COMPLEX_NAME
    kg_assumption = CWA

    def __init__(self, num_entities, num_relations, embedding_dim=200,
                 criterion=nn.BCEWithLogitsLoss(reduction='mean'), preferred_device=GPU):
        super(ComplEx, self).__init__(num_entities, num_relations, criterion, embedding_dim, preferred_device)

        self.entity_embeddings_real = nn.Embedding(self.num_entities, self.embedding_dim)
        self.entity_embeddings_img = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings_real = nn.Embedding(self.num_relations, self.embedding_dim)
        self.relation_embeddings_img = nn.Embedding(self.num_relations, self.embedding_dim)

        self.init()
        self.criterion = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.entity_embeddings_real.weight.data)
        xavier_normal_(self.entity_embeddings_img.weight.data)
        xavier_normal_(self.relation_embeddings_real.weight.data)
        xavier_normal_(self.relation_embeddings_img.weight.data)

    def _compute_loss(self, predictions, labels):
        return self.criterion(predictions, labels)

    def _score_triples(self, triples):
        heads_real, relations_real, tails_real, heads_img, relations_img, tails_img = self._get_triple_embeddings(
            triples)
        scores = self._compute_scores(heads_real, relations_real, tails_real, heads_img, relations_img, tails_img)
        return scores

    def _compute_scores(self, heads_real, relations_real, tails_real, heads_img, relations_img, tails_img):
        """."""

        # ComplEx space bilinear product (equivalent to HolE)
        # *: Elementwise multiplication
        real_real_real = torch.sum(heads_real * relations_real * tails_real, dim=1)
        real_img_img = torch.sum(heads_real * relations_img * tails_img, dim=1)
        img_real_img = torch.sum(heads_img * relations_real * tails_img, dim=1)
        img_img_real = torch.sum(heads_img * relations_img * tails_real, dim=1)

        scores = real_real_real + real_img_img + img_real_img - img_img_real

        return scores

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_embeddings(elements=heads,
                                 embedding_module=self.entity_embeddings_real,
                                 embedding_dim=self.embedding_dim),
            self._get_embeddings(elements=relations,
                                 embedding_module=self.relation_embeddings_real,
                                 embedding_dim=self.embedding_dim),
            self._get_embeddings(elements=tails,
                                 embedding_module=self.entity_embeddings_real,
                                 embedding_dim=self.embedding_dim),
            self._get_embeddings(elements=heads,
                                 embedding_module=self.entity_embeddings_img,
                                 embedding_dim=self.embedding_dim),
            self._get_embeddings(elements=relations,
                                 embedding_module=self.relation_embeddings_img,
                                 embedding_dim=self.embedding_dim),
            self._get_embeddings(elements=tails,
                                 embedding_module=self.entity_embeddings_img,
                                 embedding_dim=self.embedding_dim),
        )

    def predict_scores(self, triples):
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        """."""

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self.compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)

        return loss
