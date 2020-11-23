# -*- coding: utf-8 -*-

"""Implementation of the RotatE model."""

from typing import Optional

import torch

from ..base import ERModel
from ...losses import Loss
from ...nn import EmbeddingSpecification
from ...nn.init import init_phases, xavier_uniform_
from ...nn.modules import RotatEInteraction
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import complex_normalize

__all__ = [
    'RotatE',
]


class RotatE(ERModel):
    r"""An implementation of RotatE from [sun2019]_.

    RotatE models relations as rotations from head to tail entities in complex space:

    .. math::

        \textbf{e}_t= \textbf{e}_h \odot \textbf{r}_r

    where $\textbf{e}, \textbf{r} \in \mathbb{C}^{d}$ and the complex elements of
    $\textbf{r}_r$ are restricted to have a modulus of one ($\|\textbf{r}_r\| = 1$). The
    interaction model is then defined as:

    .. math::

        f(h,r,t) = -\|\textbf{e}_h \odot \textbf{r}_r - \textbf{e}_t\|

    which allows to model symmetry, antisymmetry, inversion, and composition.

    .. seealso::

       - Authors' `implementation of RotatE
         <https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/model.py#L200-L228>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=125, high=1000, q=100),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 200,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            interaction=RotatEInteraction(),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=xavier_uniform_,
                dtype=torch.complex64,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=init_phases,
                constrainer=complex_normalize,
                dtype=torch.complex64,
            ),
            loss=loss,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
