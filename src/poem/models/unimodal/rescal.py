# -*- coding: utf-8 -*-

"""Implementation of RESCAL."""

import torch
import torch.autograd
from torch import nn

from poem.constants import GPU, RESCAL_NAME
from poem.models.base_owa import BaseOWAModule, slice_triples

__all__ = ['RESCAL']


class RESCAL(BaseOWAModule):
    """An implementation of RESCAL [nickel2011]_.

    This model represents relations as matrices and models interactions between latent features.

    .. [nickel2011] Nickel, M., *et al.* (2011) `A Three-Way Model for Collective Learning on Multi-Relational Data
                    <http://www.cip.ifi.lmu.de/~nickel/data/slides-icml2011.pdf>`_. ICML. Vol. 11.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py
    """

    model_name = RESCAL_NAME
    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseOWAModule.hyper_params

    def __init__(self, num_entities, num_relations, embedding_dim=50,
                 criterion=nn.MarginRankingLoss(margin=1., reduction='mean'), preferred_device=GPU) -> None:
        super(RESCAL, self).__init__(num_entities, num_relations, criterion, embedding_dim, preferred_device)

        # Embeddings
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim * self.embedding_dim)

    def predict_scores(self, triples):
        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, positives, negatives):
        positive_scores = self._score_triples(positives)
        negative_scores = self._score_triples(negatives)
        loss = self.compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
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
        scores = torch.matmul(h_m_embs, t_embs).view(-1)

        # scores = torch.bmm(torch.transpose(h_emb, 1, 2), M)  # h^T M
        # scores = torch.bmm(scores, t_emb)  # (h^T M) h
        # scores = score.view(-1, 1)

        return scores

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_embeddings(elements=heads,
                                 embedding_module=self.entity_embeddings,
                                 embedding_dim=self.embedding_dim),
            self._get_embeddings(elements=relations,
                                 embedding_module=self.relation_embeddings,
                                 embedding_dim=self.embedding_dim),
            self._get_embeddings(elements=tails,
                                 embedding_module=self.entity_embeddings,
                                 embedding_dim=self.embedding_dim),
        )
