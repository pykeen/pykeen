# -*- coding: utf-8 -*-

"""Implementation of RESCAL."""

from typing import Dict

import torch
import torch.autograd
from torch import nn

from pykeen.constants import RESCAL_NAME, SCORING_FUNCTION_NORM
from pykeen.kge_models.base import BaseModule, slice_triples

__all__ = ['RESCAL']


class RESCAL(BaseModule):
    """An implementation of RESCAL [nickel2011]_.

    This model represents relations as matrices and models interactions between latent features.

    .. [nickel2011] Nickel, M., *et al.* (2011) `A Three-Way Model for Collective Learning on Multi-Relational Data
                    <http://www.cip.ifi.lmu.de/~nickel/data/slides-icml2011.pdf>`_. ICML. Vol. 11.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py
    """

    model_name = RESCAL_NAME
    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseModule.hyper_params + [SCORING_FUNCTION_NORM]

    def __init__(self, config: Dict) -> None:
        super().__init__(config)

        # Embeddings
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim * self.embedding_dim)

        self.scoring_fct_norm = config[SCORING_FUNCTION_NORM]

    def predict(self, triples):
        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, positives, negatives):
        positive_scores = self._score_triples(positives)
        negative_scores = self._score_triples(negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _score_triples(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        scores = self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings)
        return scores

    def _compute_scores(self, h_embs, r_embs, t_embs):
        # Compute score and transform result to 1D tensor
        m = r_embs.view(-1, self.embedding_dim, self.embedding_dim)
        h_embs = h_embs.unsqueeze(-1).permute([0, 2, 1])
        h_m_embs = torch.matmul(h_embs, m)
        t_embs = t_embs.unsqueeze(-1)
        scores = -torch.matmul(h_m_embs, t_embs).view(-1)

        # scores = torch.bmm(torch.transpose(h_emb, 1, 2), M)  # h^T M
        # scores = torch.bmm(scores, t_emb)  # (h^T M) h
        # scores = score.view(-1, 1)

        return scores

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails)
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)
