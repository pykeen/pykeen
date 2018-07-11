# -*- coding: utf-8 -*-
import torch
import torch.autograd
import torch.nn as nn

from utilities.constants import EMBEDDING_DIM, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS

'''Implementation based on https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransD.py'''


class TransD(nn.Module):

    def __init__(self, config):
        super(TransD, self).__init__()

        num_entities = config[NUM_ENTITIES]
        num_relations = config[NUM_RELATIONS]
        self.entity_embedding_dim = config[EMBEDDING_DIM]
        self.relation_embedding_dim = self.entity_embedding_dim
        margin_loss = config[MARGIN_LOSS]

        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.entities_embeddings = nn.Embedding(num_entities, self.entity_embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, self.relation_embedding_dim)
        self.entities_projections = nn.Embedding(num_entities, self.entity_embedding_dim)
        self.relation_projections = nn.Embedding(num_relations, self.relation_embedding_dim)
        self.margin_loss = margin_loss

    # TODO: Add normalization
    def _project_entities(self, entity_embeddings, intermediate_entity_projs, relation_projs):
        # Compute M_r
        identity_matrix = torch.eye(self.entity_embedding_dim)
        m_r = torch.sum(torch.matmul(relation_projs, torch.transpose(intermediate_entity_projs, 1, 2)), identity_matrix)
        entities_projected = torch.matmul(m_r, entity_embeddings)
        return entities_projected

    def _init(self):
        nn.init.xavier_uniform(self.entities_embeddings.weight.data)
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform(self.entities_projections.weight.data)
        nn.init.xavier_uniform(self.relation_projections.weight.data)

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
        pos_h_projs_embs = self.entities_projections(pos_heads)
        pos_r_projs_embs = self.relation_projections(pos_relations)
        pos_t_projs_embs = self.entities_projections(pos_tails)

        neg_h_embs = self.entities_embeddings(neg_heads)
        neg_r_embs = self.relation_embeddings(neg_relations)
        neg_t_embs = self.entities_embeddings(neg_tails)
        neg_h_projs_embs = self.entities_projections(neg_heads)
        neg_r_projs_embs = self.relation_projections(neg_relations)
        neg_t_projs_embs = self.entities_projections(neg_tails)

        # Project entities
        proj_pos_heads = self._project_entities(pos_h_embs, pos_h_projs_embs, pos_r_projs_embs)
        proj_pos_tails = self._project_entities(pos_t_embs, pos_t_projs_embs, pos_r_projs_embs)

        proj_neg_heads = self._project_entities(neg_h_embs, neg_h_projs_embs, neg_r_projs_embs)
        proj_neg_tails = self._project_entities(neg_t_embs, neg_t_projs_embs, neg_r_projs_embs)

        pos_score = self.compute_scores(h_embs=proj_pos_heads, r_embs=pos_r_embs, t_embs=proj_pos_tails)
        neg_score = self.compute_scores(h_embs=proj_neg_heads, r_embs=neg_r_embs, t_embs=proj_neg_tails)

        loss = self.compute_loss(pos_score=pos_score, neg_score=neg_score)

        return loss
