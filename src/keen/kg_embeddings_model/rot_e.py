# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch
import torch.autograd
import torch.nn as nn

from keen.constants import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class RotE(nn.Module):

    def __init__(self, config):
        super(RotE, self).__init__()
        self.model_name = ROT_E
        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.num_entities = config[NUM_ENTITIES]
        self.num_relations = config[NUM_RELATIONS]
        self.embedding_dim = config[EMBEDDING_DIM]
        margin_loss = config[MARGIN_LOSS]

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and config[PREFERRED_DEVICE] == GPU else CPU)

        self.entities_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.R_k_s_embeddings = nn.Embedding(self.num_relations, self.embedding_dim * self.embedding_dim)
        self.alpha_k_s = nn.Embedding(self.num_relations, 1)
        self.W_k_s = nn.Embedding(self.num_relations, self.embedding_dim * self.embedding_dim)

        self.margin_loss = margin_loss
        self.criterion = nn.MarginRankingLoss(margin=self.margin_loss, size_average=True)

        # TODO: FIXME
        self._init()

    def _init(self):
        lower_bound = -6 / np.sqrt(self.embedding_dim)
        upper_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entities_embeddings.weight.data, a=lower_bound, b=upper_bound)
        nn.init.uniform_(self.relation_embeddings.weight.data, a=lower_bound, b=upper_bound)

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

    def compute_loss(self, pos_scores, neg_scores):
        """

        :param pos_scores:
        :param neg_scores:
        :return:
        """

        # y == -1 indicates that second input to criterion should get a larger loss
        # y = torch.Tensor([-1]).cuda()
        # NOTE: y = 1 is important
        # y = torch.tensor([-1], dtype=torch.float, device=self.device)
        y = np.repeat([-1], repeats=pos_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        # Scores for the psotive and negative triples
        pos_scores = torch.tensor(pos_scores, dtype=torch.float, device=self.device)
        neg_scores = torch.tensor(neg_scores, dtype=torch.float, device=self.device)
        # neg_scores_temp = 1 * torch.tensor(neg_scores, dtype=torch.float, device=self.device)

        loss = self.criterion(pos_scores, neg_scores, y)

        return loss

    def compute_score(self, proj_head_embs, proj_tail_embs, R_k, alpha_k):
        """

        :param h_embs:
        :param r_embs:
        :param t_embs:
        :return:
        """

        # Apply rotation
        rotated_pos_head_embs = torch.mm(R_k, proj_head_embs)
        scaled_tail_embs = alpha_k * proj_tail_embs
        # Add the vector element wise
        sum_res = rotated_pos_head_embs - scaled_tail_embs
        distances = torch.norm(sum_res, dim=1).view(size=(-1,))

        return distances

    def predict(self, triples):
        """

        :param head:
        :param relation:
        :param tail:
        :return:
        """
        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)

        heads = triples[:, 0:1]
        relations = triples[:, 1:2]
        tails = triples[:, 2:3]

        head_embs = self.entities_embeddings(heads).view(-1, self.embedding_dim)
        relation_embs = self.relation_embeddings(relations).view(-1, self.embedding_dim)
        tail_embs = self.entities_embeddings(tails).view(-1, self.embedding_dim)

        scores = self.compute_score(h_embs=head_embs, r_embs=relation_embs, t_embs=tail_embs)

        return scores.detach().cpu().numpy()

    def forward(self, pos_exmpls, neg_exmpls):
        """

        :param pos_exmpls:
        :param neg_exmpls:
        :return:
        """

        # Normalise embeddings of entities
        norms = torch.norm(self.entities_embeddings.weight, p=2, dim=1).data
        self.entities_embeddings.weight.data = self.entities_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entities_embeddings.weight))

        pos_heads = pos_exmpls[:, 0:1]
        pos_relations = pos_exmpls[:, 1:2]
        pos_tails = pos_exmpls[:, 2:3]

        neg_heads = neg_exmpls[:, 0:1]
        neg_relations = neg_exmpls[:, 1:2]
        neg_tails = neg_exmpls[:, 2:3]

        pos_h_embs = self.entities_embeddings(pos_heads)

        pos_t_embs = self.entities_embeddings(pos_tails).view(-1, self.embedding_dim)

        neg_h_embs = self.entities_embeddings(neg_heads).view(-1, self.embedding_dim)

        neg_t_embs = self.entities_embeddings(neg_tails).view(-1, self.embedding_dim)

        R_k_s = self.R_k_s_embeddings(pos_relations).view(-1, self.embedding_dim,
                                                          self.embedding_dim)
        W_k_s = self.W_k_s(pos_relations).view(-1, self.embedding_dim,
                                               self.embedding_dim)

        print("W_k_s shape: ", W_k_s.shape)
        print("pos_h_embs shape: ", pos_h_embs.t().shape)
        exit(0)
        alpha_k_s = self.alpha_k_s(pos_relations)

        # Project entities into relation space
        proj_pos_heads = torch.mm(W_k_s, pos_h_embs)
        proj_pos_tails = torch.mm(W_k_s, pos_t_embs)

        proj_neg_heads = torch.mm(W_k_s, neg_h_embs)
        proj_neg_tails = torch.mm(W_k_s, neg_t_embs)

        pos_scores = self.compute_score(projected_head=proj_pos_heads, projected_tail=proj_pos_tails, R_k=R_k_s,
                                        alpha_k=alpha_k_s)
        neg_scores = self.compute_score(projected_head=proj_neg_heads, projected_tail=proj_neg_tails, R_k=R_k_s,
                                        alpha_k=alpha_k_s)

        loss = self.compute_loss(pos_scores=pos_scores, neg_scores=neg_scores)

        return loss
