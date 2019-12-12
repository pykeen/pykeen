# -*- coding: utf-8 -*-

"""Implementation of SimplE."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import BaseModule
from ...losses import Loss, SoftplusLoss
from ...regularizers import PowerSumRegularizer, Regularizer
from ...triples import TriplesFactory
from ...utils import get_embedding_in_canonical_shape, slice_triples

__all__ = [
    'SimplE',
]


class SimplE(BaseModule):
    """An implementation of SimplE [kazemi2018]_.

    This model extends CP by updating a triple, and the inverse triple.

    .. seealso::

       - Official implementation: https://github.com/Mehran-k/SimplE
       - Improved implementation in pytorch: https://github.com/baharefatemi/SimplE
    """

    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )

    criterion_default = SoftplusLoss
    criterion_default_kwargs = {}

    #: The regularizer used by [trouillon2016]_ for SimplE
    #: In the paper, they use weight of 0.1, and do not normalize the
    #: regularization term by the number of elements, which is 200.
    regularizer_default = PowerSumRegularizer
    #: The power sum settings used by [trouillon2016]_ for SimplE
    regularizer_default_kwargs = dict(
        weight=20,
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 200,
        entity_embeddings: Optional[nn.Embedding] = None,
        tail_entity_embeddings: Optional[nn.Embedding] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        inverse_relation_embeddings: Optional[nn.Embedding] = None,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.relation_embeddings = relation_embeddings
        self.tail_entity_embeddings = tail_entity_embeddings
        self.inverse_relation_embeddings = inverse_relation_embeddings

        # Finalize initialization
        self._init_weights_on_device()

    def init_empty_weights_(self):  # noqa: D102
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        if self.tail_entity_embeddings is None:
            self.tail_entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        if self.inverse_relation_embeddings is None:
            self.inverse_relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.tail_entity_embeddings = None
        self.relation_embeddings = None
        self.inverse_relation_embeddings = None
        return self

    @staticmethod
    def interaction_function(
        hh: torch.FloatTensor,
        ht: torch.FloatTensor,
        r: torch.FloatTensor,
        r_inv: torch.FloatTensor,
        th: torch.FloatTensor,
        tt: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function of SimplE for given embeddings.

        The embeddings have to be in a broadcastable shape.

        :return:
            The scores.
        """
        # Compute CP scores for triple, and inverse triple
        score = torch.sum(hh * r * tt, dim=-1)
        inverse_score = torch.sum(ht * r_inv * th, dim=-1)

        # Final score is average
        scores = 0.5 * (score + inverse_score)

        # Note: In the code in their repository, the score is clamped to [-20, 20].
        #       That is not mentioned in the paper, so it is omitted here.

        return scores

    def _score(self, h_ind: torch.LongTensor, r_ind: torch.LongTensor, t_ind: torch.LongTensor) -> torch.FloatTensor:
        # Lookup embeddings
        hh = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=h_ind)
        ht = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=t_ind)
        r = get_embedding_in_canonical_shape(embedding=self.relation_embeddings, ind=r_ind)
        r_inv = get_embedding_in_canonical_shape(embedding=self.inverse_relation_embeddings, ind=r_ind)
        th = get_embedding_in_canonical_shape(embedding=self.tail_entity_embeddings, ind=h_ind)
        tt = get_embedding_in_canonical_shape(embedding=self.tail_entity_embeddings, ind=t_ind)

        # compute scores
        scores = self.interaction_function(hh=hh, ht=ht, th=th, tt=tt, r=r, r_inv=r_inv)

        # Regularization
        self.regularize_if_necessary(hh, ht, th, tt, r, r_inv)

        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h_ind, r_ind, t_ind = slice_triples(hrt_batch)
        return self._score(h_ind=h_ind, r_ind=r_ind, t_ind=t_ind).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hr_batch[:, 0], r_ind=hr_batch[:, 1], t_ind=None)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=None, r_ind=rt_batch[:, 0], t_ind=rt_batch[:, 1])
