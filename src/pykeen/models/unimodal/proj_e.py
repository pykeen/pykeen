# -*- coding: utf-8 -*-

"""Implementation of ProjE."""

from typing import Optional

from torch import nn

from ..base import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss
from ...nn import EmbeddingSpecification
from ...nn.init import xavier_uniform_
from ...nn.modules import ProjEInteraction
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ProjE',
]


class ProjE(ERModel):
    r"""An implementation of ProjE from [shi2017]_.

    ProjE is a neural network-based approach with a *combination* and a *projection* layer. The interaction model
    first combines $h$ and $r$ by following combination operator:

    .. math::

        \textbf{h} \otimes \textbf{r} = \textbf{D}_e \textbf{h} + \textbf{D}_r \textbf{r} + \textbf{b}_c

    where $\textbf{D}_e, \textbf{D}_r \in \mathbb{R}^{k \times k}$ are diagonal matrices which are used as shared
    parameters among all entities and relations, and $\textbf{b}_c \in \mathbb{R}^{k}$ represents the candidate bias
    vector shared across all entities. Next, the score for the triple $(h,r,t) \in \mathbb{K}$ is computed:

    .. math::

        f(h, r, t) = g(\textbf{t} \ z(\textbf{h} \otimes \textbf{r}) + \textbf{b}_p)

    where $g$ and $z$ are activation functions, and $\textbf{b}_p$ represents the shared projection bias vector.

    .. seealso::

       - Official Implementation: https://github.com/nddsg/ProjE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The default loss function class
    loss_default = nn.BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(reduction='mean')

    def __init__(
        self,
        triples_factory: TriplesFactory,
        # ProjE parameters
        embedding_dim: int = 50,
        inner_non_linearity: Optional[nn.Module] = None,
        # Loss
        loss: Optional[Loss] = None,
        # Model parameters
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize :class:`ERModel` using :class:`ProjEInteraction`."""
        super().__init__(
            triples_factory=triples_factory,
            interaction=ProjEInteraction(
                embedding_dim=embedding_dim,
                inner_non_linearity=inner_non_linearity,
            ),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=xavier_uniform_,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=xavier_uniform_,
            ),
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
