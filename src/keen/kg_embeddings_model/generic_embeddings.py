# -*- coding: utf-8 -*-

import torch
import torch.autograd
import torch.nn as nn

from keen.constants import *


class GenericEmbeddings(nn.Module):

    def __init__(self, config):
        super(GenericEmbeddings, self).__init__()
        # A simple lookup table that stores embeddings of a fixed dictionary and size

        num_entities = config[NUM_ENTITIES]
        num_relations = config[NUM_RELATIONS]
        self.embedding_dim = config[EMBEDDING_DIM]
        margin_loss = config[MARGIN_LOSS]

        self.entities_embeddings = nn.Embedding(num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, self.embedding_dim)
        self.R_k_s_embeddings = nn.Embedding(num_relations, self.embedding_dim * self.embedding_dim)
        self.alpha_k_s = nn.Embedding(num_relations, 1)
        self.W_k_s = nn.Embedding(num_relations, self.embedding_dim * self.embedding_dim)
        self.margin_loss = margin_loss
        self.criterion = nn.MarginRankingLoss(margin=self.margin_loss, size_average=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, pos_exmpls, neg_exmpls):
        pos_heads = pos_exmpls[:, 0:1]
        pos_relations = pos_exmpls[:, 1:2]
        pos_tails = pos_exmpls[:, 2:3]

        neg_heads = neg_exmpls[:, 0:1]
        neg_relations = neg_exmpls[:, 1:2]
        neg_tails = neg_exmpls[:, 2:3]

        pos_h_embs = self.entities_embeddings(pos_heads)
        pos_r_embs = self.relation_embeddings(pos_relations)
        pos_t_embs = self.entities_embeddings(pos_tails)

        neg_h_embs = self.entities_embeddings(neg_heads)
        neg_r_embs = self.relation_embeddings(neg_relations)
        neg_t_embs = self.entities_embeddings(neg_tails)

        R_k_s = self.self.R_k_s_embeddings(pos_relations).view(-1, self.embedding_dim,
                                                               self.embedding_dim)
        W_k_s = self.self.W_k_s(pos_relations).view(-1, self.embedding_dim,
                                                    self.embedding_dim)
        alpha_k_s = self.alpha_k_s(pos_relations)

        # Project entities into relation space
        proj_pos_heads = self._project_entities(pos_h_embs, W_k_s)
        proj_pos_tails = self._project_entities(pos_t_embs, W_k_s)

        proj_neg_heads = self._project_entities(neg_h_embs, W_k_s)
        proj_neg_tails = self._project_entities(neg_t_embs, W_k_s)

        pos_score = self.compute_scores(projected_head=proj_pos_heads, projected_tail=proj_pos_tails, R_k=R_k_s,
                                        alpha_k=alpha_k_s)
        neg_score = self.compute_scores(projected_head=proj_neg_heads, projected_tail=proj_neg_tails, R_k=R_k_s,
                                        alpha_k=alpha_k_s)

        loss = self.compute_loss(pos_score=pos_score, neg_score=neg_score)

        return loss

    def _project_entities(self, head_emb, W_k):
        projected_head = torch.mm(W_k, head_emb)

        return projected_head

    def compute_scores(self, projected_head, projected_tail, R_k, alpha_k):
        """

        :param projected_head:
        :param r_embs:
        :param projected_tail:
        :return:
        """
        # TODO: - torch.abs(h_emb + r_emb - t_emb)
        # Compute score and transform result to 1D tensor
        # TODO: Score is the negative of the distance
        first_part = torch.mm(R_k, projected_head)
        second_part = alpha_k * projected_tail
        score = - torch.sum(torch.abs(first_part - second_part))

        return score
