# -*- coding: utf-8 -*-

"""Implementation of RESCAL."""

from torch import nn

from poem.constants import GPU, RESCAL_NAME
from poem.models.base_owa import BaseOWAModule, slice_triples

__all__ = ['RESCAL']


class RESCAL(BaseOWAModule):
    """An implementation of RESCAL [nickel2011]_.

    This model represents relations as matrices and models interactions between latent features.

    .. [nickel2011] Nickel, M., *et al.* (2011) `A Three-Way Model for Collective Learning on Multi-Relational Data
                    <http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf>`_. ICML. Vol. 11.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py
    """
    # TODO: The paper uses a regularisation term on both, the entity embeddings, as well as the relation matrices, to avoid overfitting.

    model_name = RESCAL_NAME
    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseOWAModule.hyper_params

    def __init__(self, num_entities, num_relations, embedding_dim=50,
                 criterion=nn.MarginRankingLoss(margin=1., reduction='mean'), preferred_device=GPU) -> None:
        super(RESCAL, self).__init__(num_entities, num_relations, criterion, embedding_dim, preferred_device)

        # Embeddings
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim ** 2)

    def forward(self, positives, negatives):
        positive_scores = self._score_triples(positives)
        negative_scores = self._score_triples(negatives)
        loss = self.compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _score_triples(self, triples):
        # Get triple embeddings
        heads, relations, tails = slice_triples(triples)

        # shape: (b, d)
        head_embeddings = self.entity_embeddings(heads).view(-1, 1, self.embedding_dim)
        # shape: (b, d, d)
        relation_embeddings = self.relation_embeddings(relations).view(-1, self.embedding_dim, self.embedding_dim)
        # shape: (b, d)
        tail_embeddings = self.entity_embeddings(tails).view(-1, self.embedding_dim, 1)

        scores = head_embeddings @ relation_embeddings @ tail_embeddings

        return scores
