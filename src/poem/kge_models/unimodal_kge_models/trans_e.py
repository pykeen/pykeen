# -*- coding: utf-8 -*-

"""Implementation of the TransE model."""

import logging

import numpy as np
import torch
import torch.autograd
from torch import nn

from poem.constants import TRANS_E_NAME, GPU
from poem.kge_models.base_owa import slice_triples

__all__ = [
    'TransE',
]

log = logging.getLogger(__name__)


class TransE(torch.nn.Module):
    """An implementation of TransE [borders2013]_.

     This model considers a relation as a translation from the head to the tail entity.

    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py
    """

    model_name = TRANS_E_NAME

    def __init__(self, num_entities, num_relations, embedding_dim=50, scoring_fct_norm=1, margin_loss=1,
                 preferred_device=GPU) -> None:
        super(TransE, self).__init__()

        # Embeddings

        self.scoring_fct_norm = scoring_fct_norm
        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = num_entities
        #: The number of unique relation types in the knowledge graph
        self.num_relations = num_relations
        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)

        self.relation_embeddings = nn.Embedding(num_relations, self.embedding_dim)

        self.margin_loss = margin_loss

        self.criterion = nn.MarginRankingLoss(
            margin=self.margin_loss,
            reduction='mean',
        )
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and preferred_device else 'cpu')
        self._initialize()

    def _initialize(self):
        embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

    def predict(self, triples):
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def _compute_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        y = np.repeat([-1], repeats=positive_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        loss = self.criterion(positive_scores, negative_scores, y)
        return loss

    def forward(self, batch_positives, batch_negatives):
        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=2, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _score_triples(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        scores = self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings)
        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings):
        """Compute the scores based on the head, relation, and tail embeddings.

        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: emebddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        # Add the vector element wise
        sum_res = head_embeddings + relation_embeddings - tail_embeddings
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return distances

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails),
        )

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)
