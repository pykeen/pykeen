# -*- coding: utf-8 -*-

"""Implementation of SimplE."""

from typing import Optional, Tuple, Union

import torch.autograd

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss, SoftplusLoss
from ...regularizers import PowerSumRegularizer, Regularizer
from ...triples import TriplesFactory
from ...utils import get_embedding, get_embedding_in_canonical_shape

__all__ = [
    'SimplE',
]


class SimplE(EntityRelationEmbeddingModel):
    r"""An implementation of SimplE [kazemi2018]_.

    SimplE is an extension of canonical polyadic (CP), an early tensor factorization approach in which each entity
    $e \in \mathcal{E}$ is represented by two vectors $\textbf{h}_e, \textbf{t}_e \in \mathbb{R}^d$ and each
    relation by a single vector $\textbf{r}_r \in \mathbb{R}^d$. Depending whether an entity participates in a
    triple as the head or tail entity, either $\textbf{h}$ or $\textbf{t}$ is used. Both entity
    representations are learned independently, i.e. observing a triple $(h,r,t)$, the method only updates
    $\textbf{h}_h$ and $\textbf{t}_t$. In contrast to CP, SimplE introduces for each relation $\textbf{r}_r$
    the inverse relation $\textbf{r'}_r$, and formulates its the interaction model based on both:

    .. math::

        f(h,r,t) = \frac{1}{2}\left(\left\langle\textbf{h}_h, \textbf{r}_r, \textbf{t}_t\right\rangle
        + \left\langle\textbf{h}_t, \textbf{r'}_r, \textbf{t}_h\right\rangle\right)

    Therefore, for each triple $(h,r,t) \in \mathbb{K}$, both $\textbf{h}_h$ and $\textbf{h}_t$
    as well as $\textbf{t}_h$ and $\textbf{t}_t$ are updated.

    .. seealso::

       - Official implementation: https://github.com/Mehran-k/SimplE
       - Improved implementation in pytorch: https://github.com/baharefatemi/SimplE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )
    #: The default loss function class
    loss_default = SoftplusLoss
    #: The default parameters for the default loss function class
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
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        # extra embeddings
        self.tail_entity_embeddings = get_embedding(
            num_embeddings=triples_factory.num_entities,
            embedding_dim=embedding_dim,
            device=self.device,
        )
        self.inverse_relation_embeddings = get_embedding(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=embedding_dim,
            device=self.device,
        )

        if isinstance(clamp_score, float):
            clamp_score = (-clamp_score, clamp_score)
        self.clamp = clamp_score

        # Finalize initialization
        self.reset_parameters_()

    def _reset_parameters_(self):  # noqa: D102
        for emb in [
            self.entity_embeddings,
            self.tail_entity_embeddings,
            self.relation_embeddings,
            self.inverse_relation_embeddings,
        ]:
            emb.reset_parameters()

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
