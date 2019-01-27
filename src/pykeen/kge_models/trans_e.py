# -*- coding: utf-8 -*-

"""Implementation of the TransE model."""

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.autograd
from torch import nn

from pykeen.constants import NORM_FOR_NORMALIZATION_OF_ENTITIES, SCORING_FUNCTION_NORM, TRANS_E_NAME
from pykeen.kge_models.base import BaseModule

__all__ = [
    'TransE',
    'TransEConfig',
]

log = logging.getLogger(__name__)


@dataclass
class TransEConfig:
    lp_norm: str
    scoring_function_norm: str

    @classmethod
    def from_dict(cls, config: Dict) -> 'TransEConfig':
        """Generate an instance from a dictionary."""
        return cls(
            lp_norm=config[NORM_FOR_NORMALIZATION_OF_ENTITIES],
            scoring_function_norm=config[SCORING_FUNCTION_NORM],
        )


class TransE(BaseModule):
    """An implementation of TransE [borders2013]_.

     This model considers a relation as a translation from the head to the tail entity.

    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.

    .. seealso:: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py
    """

    model_name = TRANS_E_NAME
    margin_ranking_loss_size_average: bool = True

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        config = TransEConfig.from_dict(config)

        # Embeddings
        self.l_p_norm_entities = config.lp_norm
        self.scoring_fct_norm = config.scoring_function_norm
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        self._initialize()

    def _initialize(self):
        entity_embeddings_init_bound = relation_embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=entity_embeddings_init_bound,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-relation_embeddings_init_bound,
            b=relation_embeddings_init_bound,
        )

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

    def _compute_loss(self, pos_scores, neg_scores):
        y = np.repeat([-1], repeats=pos_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        # Scores for the positive and negative triples
        pos_scores = torch.tensor(pos_scores, dtype=torch.float, device=self.device)
        neg_scores = torch.tensor(neg_scores, dtype=torch.float, device=self.device)

        loss = self.criterion(pos_scores, neg_scores, y)

        return loss

    def _compute_scores(self, h_embs, r_embs, t_embs):
        """

        :param h_embs: embeddings of head entities of dimension batchsize x embedding_dim
        :param r_embs: emebddings of relation embeddings of dimension batchsize x embedding_dim
        :param t_embs: embeddings of tail entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        # Add the vector element wise
        sum_res = h_embs + r_embs - t_embs
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))

        return distances

    def predict(self, triples):
        heads = triples[:, 0:1]
        relations = triples[:, 1:2]
        tails = triples[:, 2:3]

        head_embs = self.entity_embeddings(heads).view(-1, self.embedding_dim)
        relation_embs = self.relation_embeddings(relations).view(-1, self.embedding_dim)
        tail_embs = self.entity_embeddings(tails).view(-1, self.embedding_dim)

        scores = self._compute_scores(h_embs=head_embs, r_embs=relation_embs, t_embs=tail_embs)

        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        pos_heads = batch_positives[:, 0:1]
        pos_relations = batch_positives[:, 1:2]
        pos_tails = batch_positives[:, 2:3]

        neg_heads = batch_negatives[:, 0:1]
        neg_relations = batch_negatives[:, 1:2]
        neg_tails = batch_negatives[:, 2:3]

        pos_h_embs = self.entity_embeddings(pos_heads).view(-1, self.embedding_dim)
        pos_r_embs = self.relation_embeddings(pos_relations).view(-1, self.embedding_dim)
        pos_t_embs = self.entity_embeddings(pos_tails).view(-1, self.embedding_dim)

        neg_h_embs = self.entity_embeddings(neg_heads).view(-1, self.embedding_dim)
        neg_r_embs = self.relation_embeddings(neg_relations).view(-1, self.embedding_dim)
        neg_t_embs = self.entity_embeddings(neg_tails).view(-1, self.embedding_dim)

        pos_scores = self._compute_scores(h_embs=pos_h_embs, r_embs=pos_r_embs, t_embs=pos_t_embs)
        neg_scores = self._compute_scores(h_embs=neg_h_embs, r_embs=neg_r_embs, t_embs=neg_t_embs)

        loss = self._compute_loss(pos_scores=pos_scores, neg_scores=neg_scores)

        return loss
