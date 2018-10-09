# -*- coding: utf-8 -*-

"""Implementation of TransD."""

import numpy as np
import torch
import torch.autograd
import torch.nn as nn

from pykeen.constants import *

"""
TODO: Check, whether it makes sense to integrate identity matrices.
"""


class TransD(nn.Module):

    def __init__(self, config):
        super(TransD, self).__init__()
        self.model_name = TRANS_D_NAME
        self.num_entities = config[NUM_ENTITIES]
        self.num_relations = config[NUM_RELATIONS]
        self.entity_embedding_dim = config[EMBEDDING_DIM]
        self.relation_embedding_dim = self.entity_embedding_dim
        self.margin_loss = config[MARGIN_LOSS]
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and config[PREFERRED_DEVICE] == GPU else CPU)

        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_embedding_dim, max_norm=1)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim, max_norm=1)
        self.entity_projections = nn.Embedding(self.num_entities, self.entity_embedding_dim)
        self.relation_projections = nn.Embedding(self.num_relations, self.relation_embedding_dim)

        self.criterion = nn.MarginRankingLoss(margin=self.margin_loss, size_average=True)
        self.scoring_fct_norm = config[SCORING_FUNCTION_NORM]
        # self._initialize()

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

        loss = self.criterion(pos_scores, neg_scores, y)

        return loss

    def _project_entities(self, entity_embs, entity_proj_vecs, relation_projs):
        entity_embs = entity_embs
        relation_projs = relation_projs.unsqueeze(-1)
        entity_proj_vecs = entity_proj_vecs.unsqueeze(-1).permute([0, 2, 1])

        transfer_matrices = torch.matmul(relation_projs, entity_proj_vecs)

        projected_entity_embs = torch.einsum('nmk,nk->nm', [transfer_matrices, entity_embs])

        return projected_entity_embs

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

        h_embs = self.entity_embeddings(heads).view(-1, self.entity_embedding_dim)
        r_embs = self.relation_embeddings(relations).view(-1, self.relation_embedding_dim)
        t_embs = self.entity_embeddings(tails).view(-1, self.entity_embedding_dim)

        h_proj_vec_embs = self.entity_projections(heads).view(-1, self.entity_embedding_dim)
        r_projs_embs = self.relation_projections(relations).view(-1, self.relation_embedding_dim)
        t_proj_vec_embs = self.entity_projections(tails).view(-1, self.entity_embedding_dim)

        proj_heads = self._project_entities(h_embs, h_proj_vec_embs, r_projs_embs)
        proj_tails = self._project_entities(t_embs, t_proj_vec_embs, r_projs_embs)

        scores = self._compute_scores(h_embs=proj_heads, r_embs=r_embs, t_embs=proj_tails)

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

        pos_h_proj_vec_embs = self.entity_projections(pos_heads).view(-1, self.entity_embedding_dim)
        pos_r_projs_embs = self.relation_projections(pos_relations).view(-1, self.relation_embedding_dim)
        pos_t_proj_vec_embs = self.entity_projections(pos_tails).view(-1, self.entity_embedding_dim)

        neg_h_embs = self.entity_embeddings(neg_heads).view(-1, self.entity_embedding_dim)
        neg_r_embs = self.relation_embeddings(neg_relations).view(-1, self.relation_embedding_dim)
        neg_t_embs = self.entity_embeddings(neg_tails).view(-1, self.entity_embedding_dim)

        neg_h_proj_vec_embs = self.entity_projections(neg_heads).view(-1, self.entity_embedding_dim)
        neg_r_projs_embs = self.relation_projections(neg_relations).view(-1, self.relation_embedding_dim)
        neg_t_proj_vec_embs = self.entity_projections(neg_tails).view(-1, self.entity_embedding_dim)

        # Project entities
        proj_pos_heads = self._project_entities(pos_h_embs, pos_h_proj_vec_embs, pos_r_projs_embs)
        proj_pos_tails = self._project_entities(pos_t_embs, pos_t_proj_vec_embs, pos_r_projs_embs)

        proj_neg_heads = self._project_entities(neg_h_embs, neg_h_proj_vec_embs, neg_r_projs_embs)
        proj_neg_tails = self._project_entities(neg_t_embs, neg_t_proj_vec_embs, neg_r_projs_embs)

        pos_scores = self._compute_scores(h_embs=proj_pos_heads, r_embs=pos_r_embs, t_embs=proj_pos_tails)
        neg_scores = self._compute_scores(h_embs=proj_neg_heads, r_embs=neg_r_embs, t_embs=proj_neg_tails)

        loss = self._compute_loss(pos_scores=pos_scores, neg_scores=neg_scores)

        return loss
