# -*- coding: utf-8 -*-

"""Implementation of TransR."""

import numpy as np
import torch
import torch.autograd
import torch.nn as nn

from pykeen.constants import *

"""
Constraints: 
 * ||h||_2 <= 1: Done
 * ||r||_2 <= 1: Done
 * ||t||_2 <= 1: Done
 * ||h*M_r||_2 <= 1: TODO
 * ||t*M_r||_2 <= 1: TODO
"""


class TransR(nn.Module):

    def __init__(self, config):
        super(TransR, self).__init__()
        self.model_name = TRANS_R_NAME
        self.num_entities = config[NUM_ENTITIES]
        self.num_relations = config[NUM_RELATIONS]
        self.entity_embedding_dim = config[EMBEDDING_DIM]
        self.relation_embedding_dim = config[RELATION_EMBEDDING_DIM]
        self.margin_loss = config[MARGIN_LOSS]
        self.criterion = nn.MarginRankingLoss(margin=self.margin_loss, size_average=True)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and config[PREFERRED_DEVICE] == GPU else CPU)

        # max_norm = 1 according to the paper
        self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_embedding_dim, max_norm=1)
        # max_norm = 1 according to the paper
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim, max_norm=1)
        self.projection_matrix_embs = nn.Embedding(self.num_relations,
                                                   self.relation_embedding_dim * self.entity_embedding_dim)
        self.scoring_fct_norm = config[SCORING_FUNCTION_NORM]
        self._initialize()

    def _compute_scores(self, h_embs, r_embs, t_embs):
        """

        :param h_embs:
        :param r_embs:
        :param t_embs:
        :return:
        """

        # Add the vector element wise
        sum_res = h_embs + r_embs - t_embs
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        distances = torch.mul(distances, distances)

        return distances

    def _compute_loss(self, pos_scores, neg_scores):
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

    def _project_entities(self, entity_embs, projection_matrix_embs):
        """

        :param entity_embs:
        :param projection_matrix_embs:
        :return:
        """
        projected_entity_embs = torch.einsum('nk,nkd->nd', [entity_embs, projection_matrix_embs])

        return projected_entity_embs

    # TODO: Initilaize relation matrices as identiy matrices
    def _initialize(self):
        lower_bound = -6 / np.sqrt(self.entity_embedding_dim)
        upper_bound = 6 / np.sqrt(self.entity_embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight.data, a=lower_bound, b=upper_bound)
        nn.init.uniform_(self.relation_embeddings.weight.data, a=lower_bound, b=upper_bound)

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

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

        head_embs = self.entity_embeddings(heads).view(-1, self.entity_embedding_dim)
        relation_embs = self.relation_embeddings(relations).view(-1, self.relation_embedding_dim)
        tail_embs = self.entity_embeddings(tails).view(-1, self.entity_embedding_dim)

        proj_matrix_embs = self.projection_matrix_embs(relations).view(-1, self.entity_embedding_dim,
                                                                       self.relation_embedding_dim)

        proj_heads_embs = self._project_entities(head_embs, proj_matrix_embs)
        proj_tails_embs = self._project_entities(tail_embs, proj_matrix_embs)

        scores = self._compute_scores(h_embs=proj_heads_embs, r_embs=relation_embs, t_embs=proj_tails_embs)

        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        pos_heads = batch_positives[:, 0:1]
        pos_relations = batch_positives[:, 1:2]
        pos_tails = batch_positives[:, 2:3]

        neg_heads = batch_negatives[:, 0:1]
        neg_relations = batch_negatives[:, 1:2]
        neg_tails = batch_negatives[:, 2:3]

        pos_h_embs = self.entity_embeddings(pos_heads).view(-1, self.entity_embedding_dim)
        pos_r_embs = self.relation_embeddings(pos_relations).view(-1, self.relation_embedding_dim)
        pos_t_embs = self.entity_embeddings(pos_tails).view(-1, self.entity_embedding_dim)

        neg_h_embs = self.entity_embeddings(neg_heads).view(-1, self.entity_embedding_dim)
        neg_r_embs = self.relation_embeddings(neg_relations).view(-1, self.relation_embedding_dim)
        neg_t_embs = self.entity_embeddings(neg_tails).view(-1, self.entity_embedding_dim)

        proj_matrix_embs = self.projection_matrix_embs(pos_relations).view(-1, self.entity_embedding_dim,
                                                                           self.relation_embedding_dim)

        # Project entities into relation space
        proj_pos_heads_embs = self._project_entities(pos_h_embs, proj_matrix_embs)
        proj_pos_tails_embs = self._project_entities(pos_t_embs, proj_matrix_embs)

        proj_neg_heads_embs = self._project_entities(neg_h_embs, proj_matrix_embs)
        proj_neg_tails_embs = self._project_entities(neg_t_embs, proj_matrix_embs)

        pos_scores = self._compute_scores(h_embs=proj_pos_heads_embs, r_embs=pos_r_embs, t_embs=proj_pos_tails_embs)
        neg_scores = self._compute_scores(h_embs=proj_neg_heads_embs, r_embs=neg_r_embs, t_embs=proj_neg_tails_embs)

        loss = self._compute_loss(pos_scores=pos_scores, neg_scores=neg_scores)

        return loss
