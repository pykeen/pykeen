# -*- coding: utf-8 -*-

"""Implementation of SimplE."""

from typing import Optional, Tuple, Union

import torch.autograd
from torch import nn

from ..base import Model
from ...losses import Loss, SoftplusLoss
from ...regularizers import PowerSumRegularizer, Regularizer
from ...triples import TriplesFactory
from ...utils import get_embedding_in_canonical_shape

__all__ = [
    'SimplE',
]


class SimplE(Model):
    """An implementation of SimplE [kazemi2018]_.

    This model extends CP by updating a triple, and the inverse triple.

    .. seealso::

       - Official implementation: https://github.com/Mehran-k/SimplE
       - Improved implementation in pytorch: https://github.com/baharefatemi/SimplE
    """

    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )

    loss_default = SoftplusLoss
    loss_default_kwargs = {}

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
        automatic_memory_optimization: Optional[bool] = None,
        entity_embeddings: Optional[nn.Embedding] = None,
        tail_entity_embeddings: Optional[nn.Embedding] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        inverse_relation_embeddings: Optional[nn.Embedding] = None,
        loss: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
        clamp_score: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            entity_embeddings=entity_embeddings,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.relation_embeddings = relation_embeddings
        self.tail_entity_embeddings = tail_entity_embeddings
        self.inverse_relation_embeddings = inverse_relation_embeddings

        if isinstance(clamp_score, float):
            clamp_score = (-clamp_score, clamp_score)
        self.clamp = clamp_score

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

    def _score(self, h_ind: torch.LongTensor, r_ind: torch.LongTensor, t_ind: torch.LongTensor) -> torch.FloatTensor:
        # forward model
        h = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=h_ind)
        r = get_embedding_in_canonical_shape(embedding=self.relation_embeddings, ind=r_ind)
        t = get_embedding_in_canonical_shape(embedding=self.tail_entity_embeddings, ind=t_ind)
        scores = (h * r * t).sum(dim=-1)

        # Regularization
        self.regularize_if_necessary(h, r, t)

        # backward model
        h = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=t_ind)
        r = get_embedding_in_canonical_shape(embedding=self.inverse_relation_embeddings, ind=r_ind)
        t = get_embedding_in_canonical_shape(embedding=self.tail_entity_embeddings, ind=h_ind)
        scores = 0.5 * (scores + (h * r * t).sum(dim=-1))

        # Regularization
        self.regularize_if_necessary(h, r, t)

        # Note: In the code in their repository, the score is clamped to [-20, 20].
        #       That is not mentioned in the paper, so it is omitted here.
        if self.clamp is not None:
            min_, max_ = self.clamp
            scores = scores.clamp(min=min_, max=max_)

        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hrt_batch[:, 0], r_ind=hrt_batch[:, 1], t_ind=hrt_batch[:, 2]).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hr_batch[:, 0], r_ind=hr_batch[:, 1], t_ind=None)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=None, r_ind=rt_batch[:, 0], t_ind=rt_batch[:, 1])
