# -*- coding: utf-8 -*-
import torch
import torch.autograd
import torch.nn as nn

from utilities.constants import EMBEDDING_DIM, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS

'''Implementation based on https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransR.py'''


class TransR(nn.Module):

    def __init__(self, config):
        super(TransR, self).__init__()

        num_entities = config[NUM_ENTITIES]
        num_relations = config[NUM_RELATIONS]
        self.entity_embedding_dim = config[EMBEDDING_DIM]
        self.relation_embedding_dim = self.entity_embedding_dim
        margin_loss = config[MARGIN_LOSS]

        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.entities_embeddings = nn.Embedding(num_entities, self.entity_embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, self.entity_embedding_dim)
        # TODO: Check
        self.projection_matrices = nn.Embedding(num_relations, self.entity_embedding_dim * self.relation_embedding_dim)
        self.margin_loss = margin_loss

    def compute_score(self, h_embs, r_embs, t_embs):
        # TODO: - torch.abs(h_emb + r_emb - t_emb)
        # Compute score and transform result to 1D tensor
        score = - torch.sum(torch.abs(h_embs + r_embs - t_embs))

    def _project_entities(self, entity_embs, projection_embs):
        return torch.matmul(projection_embs, entity_embs)

    def _init(self):
        nn.init.xavier_uniform(self.entities_embeddings.weight.data)
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform(self.projection_matrices.weight.data)

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

        proj_matrix_embs = self.projection_matrices(pos_relations).view(-1, self.relation_embedding_dim,
                                                                        self.entity_embedding_dim)

        # Project entities into relation space
        proj_pos_heads = self._project_entities(pos_h_embs, proj_matrix_embs)
        proj_pos_tails = self._project_entities(pos_t_embs, proj_matrix_embs)

        proj_neg_heads = self._project_entities(neg_h_embs, proj_matrix_embs)
        proj_neg_tails = self._project_entities(neg_t_embs, proj_matrix_embs)

        pos_score = self.compute_score(h_embs=proj_neg_heads, r_embs=pos_r_embs, t_embs=pos_t_embs)
        neg_score = self.compute_score(h_embs=neg_h_embs, r_embs=neg_r_embs, t_embs=neg_t_embs)

        loss = self.compute_loss(pos_score=pos_score, neg_score=neg_score)
