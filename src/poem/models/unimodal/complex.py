# -*- coding: utf-8 -*-

"""Implementation of the Complex model based on the open world assumption (OWA)."""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from poem.constants import GPU, COMPLEX_NAME, OWA
from poem.customized_loss_functions.softplus_loss import SoftplusLoss
from poem.models.base_owa import BaseOWAModule, slice_triples


class ComplEx(BaseOWAModule):
    """An implementation of ComplEx [Trouillon2016complex].

    .. [trouillon2016complex] Trouillon, Théo, et al. "Complex embeddings for simple link prediction."
                              International Conference on Machine Learning. 2016.
    """
    model_name = COMPLEX_NAME
    kg_assumption = OWA

    def __init__(self, num_entities, num_relations, embedding_dim=200, neg_label=-1., regularization_factor=0.01,
                 criterion=SoftplusLoss(reduction='mean'), preferred_device=GPU):
        super(ComplEx, self).__init__(num_entities, num_relations, criterion, embedding_dim, preferred_device)

        self.entity_embeddings_real = self.entity_embeddings
        self.entity_embeddings_img = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings_real = nn.Embedding(self.num_relations, self.embedding_dim)
        self.relation_embeddings_img = nn.Embedding(self.num_relations, self.embedding_dim)
        self.neg_label = neg_label
        self.regularization_factor = torch.nn.Parameter(torch.Tensor([regularization_factor]))
        self.current_regularization_term = None

        self.init()
        self.criterion = criterion

    def init(self):
        xavier_normal_(self.entity_embeddings_real.weight.data)
        xavier_normal_(self.entity_embeddings_img.weight.data)
        xavier_normal_(self.relation_embeddings_real.weight.data)
        xavier_normal_(self.relation_embeddings_img.weight.data)

    def _compute_label_loss(self, pos_scores, neg_scores):
        """."""
        loss = super()._compute_label_loss(pos_elements=pos_scores, neg_elements=neg_scores)
        loss += self.regularization_factor.item() * self.current_regularization_term

        return loss

    def _score_triples(self, triples):
        heads_real, relations_real, tails_real, heads_img, relations_img, tails_img = self._get_triple_embeddings(
            triples)
        scores = self._compute_scores(heads_real, relations_real, tails_real, heads_img, relations_img, tails_img)
        return scores

    def _compute_regularization_term(self, heads_real, relations_real, tails_real, heads_img, relations_img, tails_img):
        """"""

        regularization_term = torch.norm(heads_real, dim=1, p=2).sum()
        regularization_term += torch.norm(relations_real, dim=1, p=2).sum()
        regularization_term += torch.norm(tails_real, dim=1, p=2).sum()
        regularization_term += torch.norm(heads_img, dim=1, p=2).sum()
        regularization_term += torch.norm(relations_img, dim=1, p=2).sum()
        regularization_term += torch.norm(tails_img, dim=1, p=2).sum()

        return regularization_term

    def _compute_scores(self, heads_real, relations_real, tails_real, heads_img, relations_img, tails_img):
        """."""

        # ComplEx space bilinear product (equivalent to HolE)
        # *: Elementwise multiplication
        real_real_real = heads_real * relations_real * tails_real
        real_img_img = heads_real * relations_img * tails_img
        img_real_img = heads_img * relations_real * tails_img
        img_img_real = heads_img * relations_img * tails_real

        scores = torch.sum(real_real_real + real_img_img + img_real_img - img_img_real, dim=1)

        self.current_regularization_term = self._compute_regularization_term(heads_real, relations_real, tails_real,
                                                                             heads_img, relations_img, tails_img)
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
