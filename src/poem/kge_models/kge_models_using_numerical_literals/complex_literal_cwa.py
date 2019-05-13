# -*- coding: utf-8 -*-

"""Implementation of the ComplexLiteral model based on the closed world assumption (CWA)."""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from poem.constants import INPUT_DROPOUT, COMPLEX_CWA_NAME, NUM_ENTITIES, \
    NUM_RELATIONS, EMBEDDING_DIM, PREFERRED_DEVICE, GPU, NUMERIC_LITERALS
from poem.model_config import ModelConfig


class ComplexLiteralCWA(torch.nn.Module):
    """
        An implementation of ComplexLiteral [agustinus2018] based on the closed world assumption (CWA).

        .. [agustinus2018] Kristiadi, Agustinus, et al. "Incorporating literals into knowledge graph embeddings."
                           arXiv preprint arXiv:1802.00934 (2018).
        """

    def __init__(self, num_entities, num_relations, multimodal_data, embedding_dim=50, input_dropout=0.2,
                 preferred_device=GPU):
        super(ComplexLiteralCWA, self).__init__()
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and preferred_device else 'cpu')
        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = num_entities
        #: The number of unique relation types in the knowledge graph
        self.num_relations = num_relations
        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        self.entity_embs_real = nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.entity_embs_img = nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.relation_embs_real = nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.relation_embs_img = nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)

        # Literal
        # num_ent x num_lit
        numeric_literals = multimodal_data.get(NUMERIC_LITERALS)
        self.numeric_literals = nn.Embedding.from_pretrained(
            torch.tensor(numeric_literals, dtype=torch.float, device=self.device), freeze=True)
        # Number of columns corresponds to number of literals
        self.num_of_literals = self.numeric_literals.weight.data.shape[1]

        self.real_non_lin_transf = torch.nn.Sequential(
            nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim),
            torch.nn.Tanh()
        )

        self.img_non_lin_transf = torch.nn.Sequential(
            nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim),
            torch.nn.Tanh()
        )

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.criterion = torch.nn.BCELoss()

    def init(self):
        """."""
        xavier_normal_(self.entity_embs_real.weight.data)
        xavier_normal_(self.entity_embs_img.weight.data)
        xavier_normal_(self.relation_embs_real.weight.data)
        xavier_normal_(self.relation_embs_img.weight.data)

    def _apply_g_function(self, real_embs, img_embs, literals):
        """"""
        real = self.real_non_lin_transf(torch.cat([real_embs, literals], 1))
        img = self.img_non_lin_transf(torch.cat([img_embs, literals], 1))

        return real, img

    def _compute_loss(self, predictions, labels):
        """"""
        loss = self.criterion(predictions, labels)
        return loss

    def predict(self, triples: torch.tensor):
        """."""
        heads = triples[:, 0:1]
        relations = triples[:, 1:2]
        tails = triples[:, 2:3]

        heads_embs_real = self.entity_embs_real(heads).view(-1, self.embedding_dim)
        rels_embedded_real = self.relation_embs_real(relations).view(-1, self.embedding_dim)
        tails_embs_real = self.entity_embs_real(tails).view(-1, self.embedding_dim)

        heads_embs_img = self.entity_embs_img(heads).view(-1, self.embedding_dim)
        rels_embedded_img = self.relation_embs_img(relations).view(-1, self.embedding_dim)
        tails_embs_img = self.entity_embs_img(tails).view(-1, self.embedding_dim)

        # Literals
        head_literals = self.numeric_literals(heads).view(-1, self.num_of_literals)
        tail_literals = self.numeric_literals(tails).view(-1, self.num_of_literals)

        heads_embs_real, heads_embs_img = self._apply_g_function(
            real_embs=heads_embs_real,
            img_embs=heads_embs_img,
            literals=head_literals
        )

        tails_embs_real, tails_embs_img = self._apply_g_function(
            real_embs=tails_embs_real,
            img_embs=tails_embs_img,
            literals=tail_literals
        )

        # End literals
        real_real_real = torch.bmm((heads_embs_real * rels_embedded_real).view(-1, 1, self.embedding_dim),
                                   tails_embs_real.view(-1, self.embedding_dim, 1)).view(-1)

        real_img_img = torch.bmm((heads_embs_real * rels_embedded_img).view(-1, 1, self.embedding_dim),
                                 tails_embs_img.view(-1, self.embedding_dim, 1)).view(-1)

        img_real_img = torch.bmm((heads_embs_img * heads_embs_real).view(-1, 1, self.embedding_dim),
                                tails_embs_img.view(-1, self.embedding_dim, 1)).view(-1)

        img_img_real = torch.bmm((heads_embs_img * rels_embedded_img).view(-1, 1, self.embedding_dim),
                                tails_embs_real.view(-1, self.embedding_dim, 1)).view(-1)

        predictions = real_real_real + real_img_img + img_real_img - img_img_real
        predictions = torch.sigmoid(predictions)

        return predictions.detach().cpu().numpy()

    def forward(self, batch, labels):
        """"""
        batch_heads = batch[:, 0:1]
        batch_relations = batch[:, 1:2]

        heads_embedded_real = self.inp_drop(self.entity_embs_real(batch_heads)).view(-1, self.embedding_dim)
        rels_embedded_real = self.inp_drop(self.relation_embs_real(batch_relations)).view(-1,
                                                                                          self.embedding_dim)
        heads_embedded_img = self.inp_drop(self.entity_embs_img(batch_heads)).view(-1, self.embedding_dim)
        relations_embedded_img = self.inp_drop(self.relation_embs_img(batch_relations)).view(-1,
                                                                                             self.embedding_dim)
        # Literals
        head_literals = self.numeric_literals(batch_heads).view(-1, self.num_of_literals)
        heads_embedded_real, heads_embedded_img = self._apply_g_function(
            real_embs=heads_embedded_real,
            img_embs=heads_embedded_img,
            literals=head_literals
        )

        e2_multi_emb_real = self.real_non_lin_transf(
            torch.cat([self.entity_embs_real.weight, self.numeric_literals.weight], 1))
        e2_multi_emb_img = self.img_non_lin_transf(
            torch.cat([self.entity_embs_img.weight, self.numeric_literals.weight], 1))

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
        loss = self._compute_loss(predictions=predictions, labels=labels)

        return loss
