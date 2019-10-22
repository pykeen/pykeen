# -*- coding: utf-8 -*-

"""Implementation of the ComplexLiteral model based on the closed world assumption (CWA)."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from .base_module import MultimodalBaseModule
from ...triples import TriplesNumericLiteralsFactory
from ...typing import Loss
from ...utils import slice_doubles


# TODO: Check entire build of the model
class ComplExLiteral(MultimodalBaseModule):
    """An implementation of ComplexLiteral from [agustinus2018]_ based on the closed world assumption (CWA)."""

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        input_dropout: float = 0.2,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the model."""
        if criterion is None:
            criterion = nn.BCELoss()

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        self.entity_embs_real = None
        self.entity_embs_img = None
        self.relation_embs_real = None
        self.relation_embs_img = None

        # Literal
        # num_ent x num_lit
        numeric_literals = triples_factory.numeric_literals
        self.numeric_literals = nn.Embedding.from_pretrained(
            torch.tensor(numeric_literals, dtype=torch.float, device=self.device), freeze=True,
        )
        # Number of columns corresponds to number of literals
        self.num_of_literals = self.numeric_literals.weight.data.shape[1]

        self.real_non_lin_transf = torch.nn.Sequential(
            nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim),
            torch.nn.Tanh(),
        )

        self.img_non_lin_transf = torch.nn.Sequential(
            nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim),
            torch.nn.Tanh(),
        )

        self.inp_drop = torch.nn.Dropout(input_dropout)

        self._init_embeddings()

    def _init_embeddings(self):
        self.entity_embs_real = nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.entity_embs_img = nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.relation_embs_real = nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.relation_embs_img = nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        xavier_normal_(self.entity_embs_real.weight.data)
        xavier_normal_(self.entity_embs_img.weight.data)
        xavier_normal_(self.relation_embs_real.weight.data)
        xavier_normal_(self.relation_embs_img.weight.data)

    def _apply_g_function(self, real_embs, img_embs, literals):
        real = self.real_non_lin_transf(torch.cat([real_embs, literals], 1))
        img = self.img_non_lin_transf(torch.cat([img_embs, literals], 1))
        return real, img

    def forward_cwa(self, doubles: torch.Tensor) -> torch.Tensor:
        """Forward pass using right side (object) prediction for training with the CWA."""
        batch_heads, batch_relations = slice_doubles(doubles)

        heads_embedded_real = self.inp_drop(self.entity_embs_real(batch_heads)).view(-1, self.embedding_dim)
        rels_embedded_real = self.inp_drop(self.relation_embs_real(batch_relations)).view(
            -1,
            self.embedding_dim,
        )
        heads_embedded_img = self.inp_drop(self.entity_embs_img(batch_heads)).view(-1, self.embedding_dim)
        relations_embedded_img = self.inp_drop(self.relation_embs_img(batch_relations)).view(
            -1,
            self.embedding_dim,
        )
        # Literals
        head_literals = self.numeric_literals(batch_heads).view(-1, self.num_of_literals)
        heads_embedded_real, heads_embedded_img = self._apply_g_function(
            real_embs=heads_embedded_real,
            img_embs=heads_embedded_img,
            literals=head_literals,
        )

        e2_multi_emb_real = self.real_non_lin_transf(
            torch.cat([self.entity_embs_real.weight, self.numeric_literals.weight], 1),
        )
        e2_multi_emb_img = self.img_non_lin_transf(
            torch.cat([self.entity_embs_img.weight, self.numeric_literals.weight], 1),
        )

        # End literals

        heads_embedded_real = self.inp_drop(heads_embedded_real)
        rels_embedded_real = self.inp_drop(rels_embedded_real)
        heads_embedded_img = self.inp_drop(heads_embedded_img)
        relations_embedded_img = self.inp_drop(relations_embedded_img)

        real_real_real = torch.mm(heads_embedded_real * rels_embedded_real, e2_multi_emb_real.t())
        real_img_img = torch.mm(heads_embedded_real * relations_embedded_img, e2_multi_emb_img.t())
        img_real_img = torch.mm(heads_embedded_img * heads_embedded_real, e2_multi_emb_img.t())
        img_img_real = torch.mm(heads_embedded_img * relations_embedded_img, e2_multi_emb_real.t())

        predictions = real_real_real + real_img_img + img_real_img - img_img_real
        predictions = torch.sigmoid(predictions)

        return predictions
