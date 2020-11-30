# -*- coding: utf-8 -*-

"""Implementation of SimplE."""

from typing import Any, ClassVar, Mapping, Optional, Tuple, Union

import torch

from ..base import ERModel
from ...losses import Loss, SoftplusLoss
from ...nn import EmbeddingSpecification
from ...nn.modules import SimplEInteraction
from ...regularizers import PowerSumRegularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'SimplE',
]


class SimplE(ERModel):
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
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}
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
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        clamp_score: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> None:
        regularizer = self._instantiate_default_regularizer()
        super().__init__(
            triples_factory=triples_factory,
            interaction=SimplEInteraction(clamp_score=clamp_score),
            entity_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    regularizer=regularizer,
                ),
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    regularizer=regularizer,
                ),
            ],
            relation_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    regularizer=regularizer,
                ),
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    regularizer=regularizer,
                ),
            ],
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        slice_size: Optional[int] = None,
        slice_dim: Optional[str] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        h, r, t = zip(*(
            (
                h_source.get_in_canonical_shape(dim="h", indices=h_indices),
                r_source.get_in_canonical_shape(dim="r", indices=r_indices),
                t_source.get_in_canonical_shape(dim="t", indices=t_indices),
            )
            for h_source, r_source, t_source in (
                (self.entity_representations[0], self.relation_representations[0], self.entity_representations[1]),
                (self.entity_representations[1], self.relation_representations[1], self.entity_representations[0]),
            )
        ))
        scores = self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=slice_dim)
        return self._repeat_scores_if_necessary(scores, h_indices, r_indices, t_indices)
