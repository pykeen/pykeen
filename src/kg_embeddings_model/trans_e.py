# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from utilities.constants import EMBEDDING_DIM, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, NORMALIZATION_OF_ENTITIES, \
    TRANS_E, PREFERRED_DEVICE, GPU, CPU


class TransE(nn.Module):

    def __init__(self, config):
        super(TransE, self).__init__()
        self.model_name = TRANS_E
        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.num_entities = config[NUM_ENTITIES]
        self.num_relations = config[NUM_RELATIONS]
        self.embedding_dim = config[EMBEDDING_DIM]
        margin_loss = config[MARGIN_LOSS]

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and config[PREFERRED_DEVICE] == GPU else CPU)

        self.l_p_norm = config[NORMALIZATION_OF_ENTITIES]
        self.entities_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

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

        print("pos scores: ", pos_scores)
        print("neg scores: ", neg_scores)

        loss = self.criterion(pos_scores, neg_scores, y)

        return loss

    def compute_score(self, h_embs, r_embs, t_embs):
        """

        :param h_embs:
        :param r_embs:
        :param t_embs:
        :return:
        """

        # print('h_embs: ', (h_embs))

        # Add the vector element wise
        sum_res = h_embs + r_embs - t_embs

        # Square root
        square_res = torch.mul(sum_res, sum_res).view(-1, self.embedding_dim)
        reduced_sum_res = torch.sum(square_res, 1)

        # Take the square root element wise
        sqrt_res = torch.sqrt(reduced_sum_res)
        # The scores are the distance
        distances = sqrt_res

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

        norms = torch.norm(self.entities_embeddings.weight, p=self.l_p_norm, dim=1).data
        self.entities_embeddings.weight.data = self.entities_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entities_embeddings.weight))

        pos_heads = pos_exmpls[:, 0:1]
        pos_relations = pos_exmpls[:, 1:2]
        pos_tails = pos_exmpls[:, 2:3]

        neg_heads = neg_exmpls[:, 0:1]
        neg_relations = neg_exmpls[:, 1:2]
        neg_tails = neg_exmpls[:, 2:3]

        pos_h_embs = self.entities_embeddings(pos_heads).view(-1, self.embedding_dim)
        pos_r_embs = self.relation_embeddings(pos_relations).view(-1, self.embedding_dim)

        pos_t_embs = self.entities_embeddings(pos_tails).view(-1, self.embedding_dim)

        neg_h_embs = self.entities_embeddings(neg_heads).view(-1, self.embedding_dim)
        neg_r_embs = self.relation_embeddings(neg_relations).view(-1, self.embedding_dim)
        neg_t_embs = self.entities_embeddings(neg_tails).view(-1, self.embedding_dim)

        # L-P normalization of the vectors
        # pos_h_embs = torch.nn.functional.normalize(pos_h_embs, p=self.l_p_norm, dim=1).view(-1, self.embedding_dim)
        # pos_t_embs = torch.nn.functional.normalize(pos_t_embs, p=self.l_p_norm, dim=1).view(-1, self.embedding_dim)
        # neg_h_embs = torch.nn.functional.normalize(neg_h_embs, p=self.l_p_norm, dim=1).view(-1, self.embedding_dim)
        # neg_t_embs = torch.nn.functional.normalize(neg_t_embs, p=self.l_p_norm, dim=1).view(-1, self.embedding_dim)

        pos_scores = self.compute_score(h_embs=pos_h_embs, r_embs=pos_r_embs, t_embs=pos_t_embs)
        neg_scores = self.compute_score(h_embs=neg_h_embs, r_embs=neg_r_embs, t_embs=neg_t_embs)

        print('Pos exmpls: ', (pos_exmpls))
        print('Neg exmpls: ', (neg_exmpls))

        loss = self.compute_loss(pos_scores=pos_scores, neg_scores=neg_scores)

        return loss
