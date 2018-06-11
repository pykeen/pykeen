# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd

from utilities.constants import EMBEDDING_DIM, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS

'https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransH.py'
class TransH(nn.Module):

    def __init__(self, config):
        super(TransH, self).__init__()

        num_entities = config[NUM_ENTITIES]
        num_relations = config[NUM_RELATIONS]
        embedding_dim = config[EMBEDDING_DIM]
        margin_loss = config[MARGIN_LOSS]

        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.entities_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.normal_vector_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin_loss = margin_loss

    def project_entity(self, entity_emb, normal_vec_emb):
        projection = entity_emb - (normal_vec_emb.T*entity_emb)*normal_vec_emb

        return projection

    def compute_score(self, h_emb, r_emb, t_emb):
        """

        :param h_emb:
        :param r_emb:
        :param t_emb:
        :return:
        """
        score = - (torch.sum(torch.abs(h_emb + r_emb - t_emb))**2)

        return score

    def loss_fct(self, pos_score, neg_score):
        """

        :param pos_score:
        :param neg_score:
        :return:
        """
        criterion = nn.MarginRankingLoss(margin=self.margin_loss, size_average=False)
        # y == -1 indicates that second input to criterion should get a larger loss
        y = torch.Tensor([-1])
        pos_score = pos_score.unsqueeze(0)
        neg_score = neg_score.unsqueeze(0)
        loss = criterion(pos_score, neg_score, y)

        return loss

    def predict(self, head, relation, tail):
        """

        :param head:
        :param relation:
        :param tail:
        :return:
        """
        head_emb = self.entities_embeddings(head)
        tail_emb = self.entities_embeddings(tail)
        relation_emb = self.relation_embeddings(relation)
        rel_normal_emb = self.normal_vector_embeddings(relation)

        head_emb_projected = self.project_entity(entity_emb=head_emb, normal_vec_emb=rel_normal_emb)
        tail_emb_projected = self.project_entity(entity_emb=tail_emb, normal_vec_emb=rel_normal_emb)

        score = self.compute_score(h_emb=head_emb_projected, relation_emb=relation_emb, t_emb=tail_emb_projected)

        return score

    def forward(self, pos_exmpl, neg_exmpl):
        """

        :param pos_exmpl:
        :param neg_exmpl:
        :return:
        """

        pos_head, pos_rel, pos_tail = pos_exmpl
        neg_head, neg_rel, neg_tail = neg_exmpl

        pos_head_emb = self.entities_embeddings(pos_head)
        pos_rel_emb = self.relation_embeddings(pos_rel)
        pos_tail_emb = self.entities_embeddings(pos_tail)
        pos_normal_emb = self.normal_vector_embeddings(pos_rel)

        neg_head_emb = self.entities_embeddings(neg_head)
        neg_rel_emb = self.relation_embeddings(neg_rel)
        neg_tail_emb = self.entities_embeddings(neg_tail)
        neg_normal_emb = self.normal_vector_embeddings(neg_rel)

        projected_head_pos = self.project_entity(entity_emb=pos_head_emb,normal_vec_emb=pos_normal_emb)
        projected_tail_pos = self.project_entity(entity_emb=pos_tail_emb, normal_vec_emb=pos_normal_emb)

        projected_head_neg = self.project_entity(entity_emb=neg_head_emb, normal_vec_emb=neg_normal_emb)
        projected_tail_neg = self.project_entity(entity_emb=neg_tail_emb, normal_vec_emb=neg_normal_emb)

        pos_score = self.calc_score(h_emb=projected_head_pos, r_emb=pos_rel_emb, t_emb=projected_tail_pos)
        neg_score = self.calc_score(h_emb=projected_head_pos, r_emb=neg_rel_emb, t_emb=projected_tail_neg)

        loss = self.loss_fct(pos_score=pos_score, neg_score=neg_score)

        return loss


