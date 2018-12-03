# -*- coding: utf-8 -*-

"""Implementation of TransH."""

import numpy as np
import torch
import torch.autograd
import torch.nn as nn

from pykeen.constants import *

__all__ = ['TransH']


class TransH(nn.Module):

    def __init__(self, config):
        super(TransH, self).__init__()
        self.model_name = TRANS_H_NAME
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and config[PREFERRED_DEVICE] == GPU else CPU)
        self.num_entities = config[NUM_ENTITIES]
        self.num_relations = config[NUM_RELATIONS]
        self.embedding_dim = config[EMBEDDING_DIM]
        margin_loss = config[MARGIN_LOSS]

        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.normal_vector_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.margin_loss = margin_loss
        self.weightning_soft_constraint = config[WEIGHT_SOFT_CONSTRAINT_TRANS_H]
        self.criterion = nn.MarginRankingLoss(margin=self.margin_loss, size_average=False)
        self.epsilon = torch.nn.Parameter(torch.tensor(0.005, requires_grad=True))
        self.scoring_fct_norm = config[SCORING_FUNCTION_NORM]

    def _initialize(self):
        # TODO: Add initialization
        pass

    def project_to_hyperplane(self, entity_embs, normal_vec_embs):
        """

        :param entity_embs: Embeddings of entities with shape batch_size x 1 x embedding_dimension
        :param normal_vec_embs: Normal vectors with shape batch_size x 1 x embedding_dimension
        :return: Projected entities of shape batch_size x embedding_dim
        """

        scaling_factors = torch.sum(normal_vec_embs * entity_embs, dim=-1).unsqueeze(1)
        heads_projected_on_normal_vecs = scaling_factors * normal_vec_embs
        projections = (entity_embs - heads_projected_on_normal_vecs).view(-1, self.embedding_dim)

        return projections

    def _compute_scores(self, h_embs, r_embs, t_embs):
        """

        :param h_embs:
        :param r_embs:
        :param t_embs:
        :return:
        """

        # Add the vector element wise
        sum_res = h_embs + r_embs - t_embs
        norms = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        scores = torch.mul(norms, norms)

        return scores

    def compute_soft_constraint_loss(self):
        """
        Compute the soft constraints.
        :return:
        """
        norm_of_entities = torch.norm(self.entity_embeddings.weight, p=2, dim=1)
        square_norms_entities = torch.mul(norm_of_entities, norm_of_entities)
        entity_constraint = square_norms_entities - self.num_entities * 1.
        entity_constraint = torch.abs(entity_constraint)
        entity_constraint = torch.sum(entity_constraint)

        orthogonalty_constraint_numerator = torch.mul(self.normal_vector_embeddings.weight,
                                                      self.relation_embeddings.weight)
        orthogonalty_constraint_numerator = torch.sum(orthogonalty_constraint_numerator, dim=1)
        orthogonalty_constraint_numerator = torch.mul(orthogonalty_constraint_numerator,
                                                      orthogonalty_constraint_numerator)

        orthogonalty_constraint_denominator = torch.norm(self.relation_embeddings.weight, p=2, dim=1)
        orthogonalty_constraint_denominator = torch.mul(orthogonalty_constraint_denominator,
                                                        orthogonalty_constraint_denominator)

        orthogonalty_constraint = (orthogonalty_constraint_numerator / orthogonalty_constraint_denominator) - \
                                  (self.num_relations * self.epsilon)
        orthogonalty_constraint = torch.abs(orthogonalty_constraint)
        orthogonalty_constraint = torch.sum(orthogonalty_constraint)

        soft_constraints_loss = self.weightning_soft_constraint * (entity_constraint + orthogonalty_constraint)

        return soft_constraints_loss

    def compute_loss(self, pos_scores, neg_scores):
        """

        :param pos_scores:
        :param neg_scores:
        :return:
        """

        pos_scores = torch.tensor(pos_scores, dtype=torch.float, device=self.device)
        neg_scores = torch.tensor(neg_scores, dtype=torch.float, device=self.device)

        # y == -1 indicates that second input to criterion should get a larger loss
        y = np.repeat([-1], repeats=pos_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)
        margin_ranking_loss = self.criterion(pos_scores, neg_scores, y)
        soft_constraint_loss = self.compute_soft_constraint_loss()

        loss = margin_ranking_loss + soft_constraint_loss

        return loss

    def predict(self, triples):
        """

        :param head:
        :param relation:
        :param tail:
        :return:
        """

        heads = triples[:, 0:1]
        relations = triples[:, 1:2]
        tails = triples[:, 2:3]

        head_embs = self.entity_embeddings(heads)
        tail_embs = self.entity_embeddings(tails)
        relation_embs = self.relation_embeddings(relations).view(-1, self.embedding_dim)
        normal_vec_embs = self.normal_vector_embeddings(relations)

        head_embs_projected = self.project_to_hyperplane(entity_embs=head_embs, normal_vec_embs=normal_vec_embs)
        tail_embs_projected = self.project_to_hyperplane(entity_embs=tail_embs, normal_vec_embs=normal_vec_embs)

        scores = self._compute_scores(h_embs=head_embs_projected, r_embs=relation_embs, t_embs=tail_embs_projected)

        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        """

        :param batch_positives:
        :param batch_negatives:
        :return:
        """

        # Normalise the normal vectors by their l2 norms
        norms = torch.norm(self.normal_vector_embeddings.weight, p=2, dim=1).data
        self.normal_vector_embeddings.weight.data = self.normal_vector_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.normal_vector_embeddings.weight))

        pos_heads = batch_positives[:, 0:1]
        pos_rels = batch_positives[:, 1:2]
        pos_tails = batch_positives[:, 2:3]

        neg_heads = batch_negatives[:, 0:1]
        neg_rels = batch_negatives[:, 1:2]
        neg_tails = batch_negatives[:, 2:3]

        # Shape: (batch_size, 1, embedding_dimension)
        pos_head_embs = self.entity_embeddings(pos_heads)
        # Reshape relation embeddings to the same shape of the projected entities
        pos_rel_embs = self.relation_embeddings(pos_rels).view(-1, self.embedding_dim)
        pos_tail_embs = self.entity_embeddings(pos_tails)
        pos_normal_embs = self.normal_vector_embeddings(pos_rels)

        neg_head_embs = self.entity_embeddings(neg_heads)
        # Reshape relation embeddings to the same shape of the projected entities
        neg_rel_embs = self.relation_embeddings(neg_rels).view(-1, self.embedding_dim)
        neg_tail_embs = self.entity_embeddings(neg_tails)
        neg_normal_embs = self.normal_vector_embeddings(neg_rels)

        projected_heads_pos = self.project_to_hyperplane(entity_embs=pos_head_embs, normal_vec_embs=pos_normal_embs)
        projected_tails_pos = self.project_to_hyperplane(entity_embs=pos_tail_embs, normal_vec_embs=pos_normal_embs)

        projected_heads_neg = self.project_to_hyperplane(entity_embs=neg_head_embs, normal_vec_embs=neg_normal_embs)
        projected_tails_neg = self.project_to_hyperplane(entity_embs=neg_tail_embs, normal_vec_embs=neg_normal_embs)

        pos_scores = self._compute_scores(h_embs=projected_heads_pos, r_embs=pos_rel_embs, t_embs=projected_tails_pos)
        neg_scores = self._compute_scores(h_embs=projected_heads_neg, r_embs=neg_rel_embs, t_embs=projected_tails_neg)

        loss = self.compute_loss(pos_scores=pos_scores, neg_scores=neg_scores)

        return loss
