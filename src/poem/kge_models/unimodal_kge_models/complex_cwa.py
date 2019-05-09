# -*- coding: utf-8 -*-

"""Implementation of the Complex model based on the closed world assumption (CWA)."""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from poem.constants import COMPLEX_CWA_NAME, GPU, CWA


class ComplexCWA(torch.nn.Module):
    """
       An implementation of Complex [agustinus2018] based on the closed world assumption (CWA)

       .. [trouillon2016complex] Trouillon, Théo, et al. "Complex embeddings for simple link prediction."
                                 International Conference on Machine Learning. 2016..
    """
    model_name = COMPLEX_CWA_NAME
    kg_assumption = CWA

    def __init__(self, num_entities, num_relations, embedding_dim=50, input_dropout=0.2, preferred_device=GPU):
        super(ComplexCWA, self).__init__()
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and preferred_device else 'cpu')
        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = num_entities
        #: The number of unique relation types in the knowledge graph
        self.num_relations = num_relations
        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        # ToDo:Why padding?
        self.entity_embeddings_real = nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.entity_embeddings_img = nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.relation_embeddings_real = nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.relation_embeddings_img = nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)

        self.init()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.criterion = torch.nn.BCELoss()

    def init(self):
        """."""
        xavier_normal_(self.entity_embeddings_real.weight.data)
        xavier_normal_(self.entity_embeddings_img.weight.data)
        xavier_normal_(self.relation_embeddings_real.weight.data)
        xavier_normal_(self.relation_embeddings_img.weight.data)

    def _compute_loss(self, predictions, labels):
        """"""
        loss = self.criterion(predictions, labels)
        return loss

    def forward(self, batch, labels):
        """"""
        batch_heads = batch[:, 0:1]
        batch_relations = batch[:, 1:2]

        subjects_embedded_real = self.inp_drop(self.entity_embeddings_real(batch_heads)).view(-1, self.embedding_dim)
        relations_embedded_real = self.inp_drop(self.relation_embeddings_real(batch_relations)).view(-1,
                                                                                                     self.embedding_dim)
        subjects_embedded_img = self.inp_drop(self.entity_embeddings_img(batch_heads)).view(-1, self.embedding_dim)
        relations_embedded_img = self.inp_drop(self.relation_embeddings_img(batch_relations)).view(-1,
                                                                                                   self.embedding_dim)

        # Apply dropout
        subjects_embedded_real = self.inp_drop(subjects_embedded_real)
        relations_embedded_real = self.inp_drop(relations_embedded_real)
        subjects_embedded_img = self.inp_drop(subjects_embedded_img)
        relations_embedded_img = self.inp_drop(relations_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        # *: Elementwise multiplication; torch.mm: matrix multiplication (does not broadcast)
        real_real_real = torch.mm(subjects_embedded_real * relations_embedded_real,
                                  self.entity_embeddings_real.weight.transpose(1, 0))
        real_img_img = torch.mm(subjects_embedded_real * relations_embedded_img,
                                self.entity_embeddings_img.weight.transpose(1, 0))
        img_real_img = torch.mm(subjects_embedded_img * relations_embedded_real,
                                self.entity_embeddings_img.weight.transpose(1, 0))
        img_img_real = torch.mm(subjects_embedded_img * relations_embedded_img,
                                self.entity_embeddings_real.weight.transpose(1, 0))

        predictions = real_real_real + real_img_img + img_real_img - img_img_real
        predictions = torch.sigmoid(predictions)
        loss = self._compute_loss(predictions=predictions, labels=labels)
        return loss
