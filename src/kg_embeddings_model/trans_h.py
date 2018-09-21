# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.autograd
import torch.nn as nn

from utilities.constants import EMBEDDING_DIM, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, TRANS_H, \
    WEIGHT_SOFT_CONSTRAINT_TRANS_H

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
        self.weightning_soft_constraint = torch.tensor([config[WEIGHT_SOFT_CONSTRAINT_TRANS_H]], dtype=torch.float,
                                                       requires_grad=True)
        self.epsilon = 0.05

    def _initialize(self):
        pass

    def project_to_hyperplane(self, entity_emb, normal_vec_emb):
        projection = entity_emb - (normal_vec_emb.T * entity_emb) * normal_vec_emb

        return projection

    def compute_score(self, h_embs, r_embs, t_embs):
        """

        :param h_embs:
        :param r_embs:
        :param t_embs:
        :return:
        """

        # Add the vector element wise
        sum_res = h_embs + r_embs - t_embs
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))

        return distances

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
        orthogonalty_constraint = (orthogonalty_constraint_numerator / orthogonalty_constraint_denominator) - (
                self.num_relations * self.epsilon)
        orthogonalty_constraint = torch.abs(orthogonalty_constraint)
        orthogonalty_constraint = torch.sum(orthogonalty_constraint)
        soft_constraints = self.weightning_soft_constraint * (entity_constraint + orthogonalty_constraint)

        loss = margin_ranking_loss + soft_constraints

        return loss

    def predict(self, triples):
        """

        :param head:
        :param relation:
        :param tail:
        :return:
        """

        heads, relations, tails = triples
        head_embs = self.entity_embeddings(heads)
        tail_embs = self.entity_embeddings(tails)
        relation_embs = self.relation_embeddings(relations)
        normal_vec_embs = self.normal_vector_embeddings(relations)

        head_emb_projected = self.project_to_hyperplane(entity_emb=head_embs, normal_vec_emb=normal_vec_embs)
        tail_emb_projected = self.project_to_hyperplane(entity_emb=tail_embs, normal_vec_emb=normal_vec_embs)

        scores = self.compute_score(h_emb=head_emb_projected, relation_emb=relation_embs, t_emb=tail_emb_projected)

        return scores

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

        projected_heads_pos = self.project_to_hyperplane(entity_emb=pos_head_emb, normal_vec_emb=pos_normal_emb)
        projected_tails_pos = self.project_to_hyperplane(entity_emb=pos_tail_emb, normal_vec_emb=pos_normal_emb)

        projected_heads_neg = self.project_to_hyperplane(entity_emb=neg_head_emb, normal_vec_emb=neg_normal_emb)
        projected_tails_neg = self.project_to_hyperplane(entity_emb=neg_tail_emb, normal_vec_emb=neg_normal_emb)

        pos_scores = self.calc_score(h_emb=projected_heads_pos, r_emb=pos_rel_emb, t_emb=projected_tails_pos)
        neg_scores = self.calc_score(h_emb=projected_heads_neg, r_emb=neg_rel_emb, t_emb=projected_tails_neg)

        loss = self.compute_loss(pos_scores=pos_scores, neg_scores=neg_scores)

        return loss
