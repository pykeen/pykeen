# -*- coding: utf-8 -*-

"""Implementation of the Complex model based on the closed world assumption (CWA)."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from poem.instance_creation_factories.triples_factory import TriplesFactory
from poem.models.base import BaseModule
from poem.utils import slice_doubles, slice_triples
from ...typing import OptionalLoss


# TODO: Combine with the Complex Module
class ComplexCWA(BaseModule):
    """An implementation of Complex [trouillon2016]_ based on the closed world assumption (CWA)."""

    def __init__(
            self,
            triples_factory: TriplesFactory,
            embedding_dim: int = 50,
            input_dropout: float = 0.2,
            criterion: OptionalLoss = None,
            preferred_device: Optional[str] = None,
            random_seed: Optional[int] = None,
    ) -> None:
        if criterion is None:
            criterion = torch.nn.BCELoss()

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
        self.inp_drop = torch.nn.Dropout(input_dropout)

        # The embeddings are first initialized when calling the get_grad_params function
        self.entity_embeddings_real = None
        self.entity_embeddings_img = None
        self.relation_embeddings_real = None
        self.relation_embeddings_img = None

        self._init_embeddings()

    def _init_embeddings(self):
        # TODO Why padding?
        self.entity_embeddings_real = nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.entity_embeddings_img = nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.relation_embeddings_real = nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.relation_embeddings_img = nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        xavier_normal_(self.entity_embeddings_real.weight.data)
        xavier_normal_(self.entity_embeddings_img.weight.data)
        xavier_normal_(self.relation_embeddings_real.weight.data)
        xavier_normal_(self.relation_embeddings_img.weight.data)

    def forward_cwa(self, doubles):
        batch_heads, batch_relations = slice_doubles(doubles)

        subjects_embedded_real = self.entity_embeddings_real(batch_heads).view(-1, self.embedding_dim)
        relations_embedded_real = self.relation_embeddings_real(batch_relations).view(-1, self.embedding_dim)
        subjects_embedded_img = self.entity_embeddings_img(batch_heads).view(-1, self.embedding_dim)
        relations_embedded_img = self.relation_embeddings_img(batch_relations).view(-1, self.embedding_dim)

        # Apply dropout
        subjects_embedded_real = self.inp_drop(subjects_embedded_real)
        relations_embedded_real = self.inp_drop(relations_embedded_real)
        subjects_embedded_img = self.inp_drop(subjects_embedded_img)
        relations_embedded_img = self.inp_drop(relations_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        # *: Elementwise multiplication; torch.mm: matrix multiplication (does not broadcast)
        real_real_real = torch.mm(
            subjects_embedded_real * relations_embedded_real,
            self.entity_embeddings_real.weight.transpose(1, 0),
        )
        real_img_img = torch.mm(
            subjects_embedded_real * relations_embedded_img,
            self.entity_embeddings_img.weight.transpose(1, 0),
        )
        img_real_img = torch.mm(
            subjects_embedded_img * relations_embedded_real,
            self.entity_embeddings_img.weight.transpose(1, 0),
        )
        img_img_real = torch.mm(
            subjects_embedded_img * relations_embedded_img,
            self.entity_embeddings_real.weight.transpose(1, 0),
        )

        predictions = real_real_real + real_img_img + img_real_img - img_img_real
        predictions = torch.sigmoid(predictions)
        return predictions

    def forward_owa(self, triples):
        batch_heads, batch_relations, batch_tails = slice_triples(triples)

        subjects_embedded_real = self.entity_embeddings_real(batch_heads).view(-1, self.embedding_dim)
        relations_embedded_real = self.relation_embeddings_real(batch_relations).view(-1, self.embedding_dim)
        objects_embedded_real = self.entity_embeddings_real(batch_tails).view(-1, self.embedding_dim)

        subjects_embedded_img = self.entity_embeddings_img(batch_heads).view(-1, self.embedding_dim)
        relations_embedded_img = self.relation_embeddings_img(batch_relations).view(-1, self.embedding_dim)
        objects_embedded_img = self.entity_embeddings_img(batch_tails).view(-1, self.embedding_dim)

        # Apply dropout
        subjects_embedded_real = self.inp_drop(subjects_embedded_real)
        relations_embedded_real = self.inp_drop(relations_embedded_real)
        subjects_embedded_img = self.inp_drop(subjects_embedded_img)
        relations_embedded_img = self.inp_drop(relations_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        # *: Elementwise multiplication; torch.mm: matrix multiplication (does not broadcast)
        real_real_real = torch.mm(
            subjects_embedded_real * relations_embedded_real,
            objects_embedded_real.transpose(1, 0),
        )
        real_img_img = torch.mm(
            subjects_embedded_real * relations_embedded_img,
            objects_embedded_img.transpose(1, 0),
        )
        img_real_img = torch.mm(
            subjects_embedded_img * relations_embedded_real,
            objects_embedded_img.transpose(1, 0),
        )
        img_img_real = torch.mm(
            subjects_embedded_img * relations_embedded_img,
            objects_embedded_real.transpose(1, 0),
        )

        predictions = real_real_real + real_img_img + img_real_img - img_img_real
        predictions = torch.sigmoid(predictions)
        return predictions
