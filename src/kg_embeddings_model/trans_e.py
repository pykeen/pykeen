# -*- coding: utf-8 -*-
import torch
import torch.autograd
import torch.nn as nn

from utilities.constants import EMBEDDING_DIM, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS


class TransE(nn.Module):

    def __init__(self, config):
        super(TransE, self).__init__()
        # A simple lookup table that stores embeddings of a fixed dictionary and size

        num_entities = config[NUM_ENTITIES]
        num_relations = config[NUM_RELATIONS]
        embedding_dim = config[EMBEDDING_DIM]
        margin_loss = config[MARGIN_LOSS]

        self.entities_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin_loss = margin_loss
        self.criterion = nn.MarginRankingLoss(margin=self.margin_loss, size_average=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_loss(self, pos_score, neg_score):
        """

        :param pos_score:
        :param neg_score:
        :return:
        """

        # y == -1 indicates that second input to criterion should get a larger loss
        # y = torch.Tensor([-1]).cuda()
        y = torch.tensor([1], dtype=torch.float, device=self.device)
        pos_score = pos_score.unsqueeze(0)
        neg_score = neg_score.unsqueeze(0)
        pos_score = torch.tensor(pos_score, dtype=torch.float, device=self.device)
        neg_score = torch.tensor(neg_score, dtype=torch.float, device=self.device)
        loss = self.criterion(pos_score, neg_score, y)

        return loss

    def calc_score(self, h_embs, r_embs, t_embs):
        """

        :param h_embs:
        :param r_embs:
        :param t_embs:
        :return:
        """
        # TODO: - torch.abs(h_emb + r_emb - t_emb)
        # Compute score and transform result to 1D tensor
        score = - torch.sum(torch.abs(h_embs + r_embs - t_embs))

        return score

    def predict(self, triple):
        """

        :param head:
        :param relation:
        :param tail:
        :return:
        """
        triple = torch.tensor(triple, dtype=torch.long, device=self.device)
        head, relation, tail = triple

        head_emb = self.entities_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entities_embeddings(tail)

        score = self.calc_score(h_embs=head_emb, r_embs=relation_emb, t_embs=tail_emb)

        return score.detach().cpu().numpy()

    def forward(self, pos_exmpl, neg_exmpl):
        """

        :param pos_exmpl:
        :param neg_exmpl:
        :return:
        """

        pos_heads = pos_exmpl[:,0:1]
        pos_relations = pos_exmpl[:,1:2]
        pos_tails =  pos_exmpl[:,2:3]

        neg_heads = neg_exmpl[:, 0:1]
        neg_relations = neg_exmpl[:, 1:2]
        neg_tails = neg_exmpl[:, 2:3]

        pos_h_embs = self.entities_embeddings(pos_heads)
        pos_r_embs = self.relation_embeddings(pos_relations)
        pos_t_embs = self.entities_embeddings(pos_tails)

        neg_h_embs = self.entities_embeddings(neg_heads)
        neg_r_embs = self.relation_embeddings(neg_relations)
        neg_t_embs = self.entities_embeddings(neg_tails)


        pos_score = self.calc_score(h_embs=pos_h_embs, r_embs=pos_r_embs, t_embs=pos_t_embs)
        neg_score = self.calc_score(h_embs=neg_h_embs, r_embs=neg_r_embs, t_embs=neg_t_embs)


        loss = self.compute_loss(pos_score=pos_score, neg_score=neg_score)



        return loss
