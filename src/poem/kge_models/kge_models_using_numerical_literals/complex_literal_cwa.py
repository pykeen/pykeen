# -*- coding: utf-8 -*-

"""Implementation of the ComplexLiteral model based on the closed world assumption (CWA)."""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from poem.constants import INPUT_DROPOUT, COMPLEX_CWA_NAME, NUM_ENTITIES, \
    NUM_RELATIONS, EMBEDDING_DIM, PREFERRED_DEVICE
from poem.model_config import ModelConfig


class ComplexLiteral(torch.nn.Module):
    """
        An implementation of ComplexLiteral [agustinus2018] based on the closed world assumption (CWA).

        .. [agustinus2018] Kristiadi, Agustinus, et al. "Incorporating literals into knowledge graph embeddings."
                           arXiv preprint arXiv:1802.00934 (2018).
        """

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(ComplexLiteral, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e_real = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit_real = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        self.emb_num_lit_img = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        e1_emb_real = self.emb_e_real(e1).view(Config.batch_size, -1)
        rel_emb_real = self.emb_rel_real(rel).view(Config.batch_size, -1)
        e1_emb_img = self.emb_e_img(e1).view(Config.batch_size, -1)
        rel_emb_img = self.emb_rel_img(rel).view(Config.batch_size, -1)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_emb_real = self.emb_num_lit_real(torch.cat([e1_emb_real, e1_num_lit], 1))
        e1_emb_img = self.emb_num_lit_img(torch.cat([e1_emb_img, e1_num_lit], 1))

        e2_multi_emb_real = self.emb_num_lit_real(torch.cat([self.emb_e_real.weight, self.numerical_literals], 1))
        e2_multi_emb_img = self.emb_num_lit_img(torch.cat([self.emb_e_img.weight, self.numerical_literals], 1))

        # End literals

        e1_emb_real = self.inp_drop(e1_emb_real)
        rel_emb_real = self.inp_drop(rel_emb_real)
        e1_emb_img = self.inp_drop(e1_emb_img)
        rel_emb_img = self.inp_drop(rel_emb_img)

        realrealreal = torch.mm(e1_emb_real*rel_emb_real, e2_multi_emb_real.t())
        realimgimg = torch.mm(e1_emb_real*rel_emb_img, e2_multi_emb_img.t())
        imgrealimg = torch.mm(e1_emb_img*rel_emb_real, e2_multi_emb_img.t())
        imgimgreal = torch.mm(e1_emb_img*rel_emb_img, e2_multi_emb_real.t())

        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred