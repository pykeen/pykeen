# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.autograd
import torch.nn as nn

from utilities.constants import EMBEDDING_DIM, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, TRANS_H

'Based on https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransH.py'


class TransH(nn.Module):

    def __init__(self, config):
        super(TransH, self).__init__()
        self.model_name = TRANS_H
        self.num_entities = config[NUM_ENTITIES]
        self.num_relations = config[NUM_RELATIONS]
        embedding_dim = config[EMBEDDING_DIM]
        margin_loss = config[MARGIN_LOSS]

        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.entity_embeddings = nn.Embedding(self.num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, embedding_dim)
        self.projected_relation_embeddings = nn.Embedding(self.num_relations, embedding_dim)
        self.normal_vector_embeddings = nn.Embedding(self.num_relations, embedding_dim)
        self.margin_loss = margin_loss
        self.c = torch.tensor(requires_grad=True)
        self.epsilon = 0.05

    def _initialize(self):
        pass

    def project_to_hyperplane(self, entity_emb, normal_vec_emb):
        projection = entity_emb - (normal_vec_emb.T * entity_emb) * normal_vec_emb

        return projection

    def compute_score(self, h_emb, r_emb, t_emb):
        """

        :param h_emb:
        :param r_emb:
        :param t_emb:
        :return:
        """
        score = - (torch.sum(torch.abs(h_emb + r_emb - t_emb)) ** 2)

        return score

    def compute_loss(self, pos_scores, neg_scores):
        """

        :param pos_scores:
        :param neg_scores:
        :return:
        """
        criterion = nn.MarginRankingLoss(margin=self.margin_loss, size_average=False)
        # y == -1 indicates that second input to criterion should get a larger loss
        y = np.repeat([-1], repeats=pos_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)
        margin_ranking_loss = criterion(pos_scores, neg_scores, y)

        norm_of_entities = torch.norm(self.entity_embeddings.weight, p=2, dim=1)
        square_norms_entities = torch.mul(norm_of_entities, norm_of_entities)
        entity_constraint = square_norms_entities - self.num_entities * 1.
        entity_constraint = torch.abs(entity_constraint)
        entity_constraint = torch.sum(entity_constraint)

        orthogonalty_constraint_numerator = torch.mul(self.normal_vector_embeddings,
                                                      self.projected_relation_embeddings.weight)
        orthogonalty_constraint_numerator = torch.sum(orthogonalty_constraint_numerator, dim=1)
        orthogonalty_constraint_numerator = torch.mul(orthogonalty_constraint_numerator,
                                                      orthogonalty_constraint_numerator)

        orthogonalty_constraint_denominator = torch.norm(self.projected_relation_embeddings.weight, p=2, dim=1)
        orthogonalty_constraint_denominator = torch.mul(orthogonalty_constraint_denominator,
                                                        orthogonalty_constraint_denominator)
        relation_constraint = (orthogonalty_constraint_numerator / orthogonalty_constraint_denominator) - (
                self.num_relations * self.epsilon)
        relation_constraint = torch.abs(relation_constraint)
        relation_constraint = torch.sum(relation_constraint)
        soft_constraints = self.c * (entity_constraint + relation_constraint)

        loss = margin_ranking_loss + soft_constraints

        return loss

    def predict(self, head, relation, tail):
        """

        :param head:
        :param relation:
        :param tail:
        :return:
        """
        head_emb = self.entity_embeddings(head)
        tail_emb = self.entity_embeddings(tail)
        relation_emb = self.relation_embeddings(relation)
        rel_normal_emb = self.normal_vector_embeddings(relation)

        head_emb_projected = self.project_to_hyperplane(entity_emb=head_emb, normal_vec_emb=rel_normal_emb)
        tail_emb_projected = self.project_to_hyperplane(entity_emb=tail_emb, normal_vec_emb=rel_normal_emb)

        score = self.compute_score(h_emb=head_emb_projected, relation_emb=relation_emb, t_emb=tail_emb_projected)

        return score

    def forward(self, batch_positives, batch_negatives):
        """

        :param batch_positives:
        :param batch_negatives:
        :return:
        """

        # Normalise embeddings of normal vectors
        norms = torch.norm(self.normal_vector_embeddings.weight, p=2, dim=1).data
        self.self.normal_vector_embeddings.weight.data = self.self.normal_vector_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.self.normal_vector_embeddings.weight))

        # TODO: Check indexing
        pos_heads, pos_rels, pos_tails = batch_positives
        neg_head, neg_rel, neg_tail = batch_negatives

        pos_head_emb = self.entity_embeddings(pos_heads)
        pos_rel_emb = self.relation_embeddings(pos_rels)
        pos_tail_emb = self.entity_embeddings(pos_tails)
        pos_normal_emb = self.normal_vector_embeddings(pos_rels)

        neg_head_emb = self.entity_embeddings(neg_head)
        neg_rel_emb = self.relation_embeddings(neg_rel)
        neg_tail_emb = self.entity_embeddings(neg_tail)
        neg_normal_emb = self.normal_vector_embeddings(neg_rel)

        projected_head_pos = self.project_to_hyperplane(entity_emb=pos_head_emb, normal_vec_emb=pos_normal_emb)
        projected_tail_pos = self.project_to_hyperplane(entity_emb=pos_tail_emb, normal_vec_emb=pos_normal_emb)

        projected_head_neg = self.project_to_hyperplane(entity_emb=neg_head_emb, normal_vec_emb=neg_normal_emb)
        projected_tail_neg = self.project_to_hyperplane(entity_emb=neg_tail_emb, normal_vec_emb=neg_normal_emb)

        pos_score = self.calc_score(h_emb=projected_head_pos, r_emb=pos_rel_emb, t_emb=projected_tail_pos)
        neg_score = self.calc_score(h_emb=projected_head_neg, r_emb=neg_rel_emb, t_emb=projected_tail_neg)

        loss = self.compute_loss(pos_scores=pos_score, neg_scores=neg_score)

        return loss
