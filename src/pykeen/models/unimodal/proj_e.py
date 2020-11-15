# -*- coding: utf-8 -*-

"""Implementation of ProjE."""

from typing import Optional

from torch import nn

from .. import SingleVectorEmbeddingModel
from ...losses import Loss
from ...nn.emb import EmbeddingSpecification
from ...nn.init import xavier_uniform_
from ...nn.modules import ProjEInteraction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ProjE',
]


class ProjE(SingleVectorEmbeddingModel):
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
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )
    #: The default loss function class
    loss_default = nn.BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(reduction='mean')

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        inner_non_linearity: Optional[nn.Module] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            interaction_function=ProjEInteraction(
                embedding_dim=embedding_dim,
                inner_non_linearity=inner_non_linearity,
            ),
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            embedding_specification=EmbeddingSpecification(
                initializer=xavier_uniform_,
            ),
            relation_embedding_specification=EmbeddingSpecification(
                initializer=xavier_uniform_,
            ),
        )
