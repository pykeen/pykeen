# -*- coding: utf-8 -*-

"""Implementation of SimplE."""

from typing import Any, ClassVar, Mapping, Optional, Tuple, Type, Union

import torch.autograd

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss, SoftplusLoss
from ...nn.emb import Embedding, EmbeddingSpecification
from ...regularizers import PowerSumRegularizer, Regularizer
from ...typing import Hint, Initializer

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
    ---
    citation:
        author: Kazemi
        year: 2018
        link: https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs
        github: Mehran-k/SimplE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = SoftplusLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}
    #: The regularizer used by [trouillon2016]_ for SimplE
    #: In the paper, they use weight of 0.1, and do not normalize the
    #: regularization term by the number of elements, which is 200.
    regularizer_default: ClassVar[Type[Regularizer]] = PowerSumRegularizer
    #: The power sum settings used by [trouillon2016]_ for SimplE
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=20,
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        clamp_score: Optional[Union[float, Tuple[float, float]]] = None,
        entity_initializer: Hint[Initializer] = None,
        relation_initializer: Hint[Initializer] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
        )

        # extra embeddings
        self.tail_entity_embeddings = Embedding.init_with_device(
            num_embeddings=self.num_entities,
            embedding_dim=embedding_dim,
            device=self.device,
        )
        self.inverse_relation_embeddings = Embedding.init_with_device(
            num_embeddings=self.num_relations,
            embedding_dim=embedding_dim,
            device=self.device,
        )

        if isinstance(clamp_score, float):
            clamp_score = (-clamp_score, clamp_score)
        self.clamp = clamp_score

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        for emb in [
            self.tail_entity_embeddings,
            self.inverse_relation_embeddings,
        ]:
            emb.reset_parameters()

    def _score(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:  # noqa: D102
        # forward model
        h = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        r = self.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        t = self.tail_entity_embeddings.get_in_canonical_shape(indices=t_indices)
        scores = (h * r * t).sum(dim=-1)

        # Regularization
        self.regularize_if_necessary(h, r, t)

        # backward model
        h = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)
        r = self.inverse_relation_embeddings.get_in_canonical_shape(indices=r_indices)
        t = self.tail_entity_embeddings.get_in_canonical_shape(indices=h_indices)
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
        return self._score(h_indices=hrt_batch[:, 0], r_indices=hrt_batch[:, 1], t_indices=hrt_batch[:, 2]).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_indices=hr_batch[:, 0], r_indices=hr_batch[:, 1], t_indices=None)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_indices=None, r_indices=rt_batch[:, 0], t_indices=rt_batch[:, 1])
