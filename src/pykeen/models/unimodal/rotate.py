# -*- coding: utf-8 -*-

"""Implementation of the RotatE model."""

from typing import Optional

import numpy as np
import torch
import torch.autograd
from torch.nn import functional

from .. import SingleVectorEmbeddingModel
from ...losses import Loss
from ...nn.emb import EmbeddingSpecification
from ...nn.init import xavier_uniform_
from ...nn.modules import RotatEInteraction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'RotatE',
]


def init_phases(x: torch.Tensor) -> torch.Tensor:
    r"""Generate random phases between 0 and :math:`2\pi`."""
    phases = 2 * np.pi * torch.rand_like(x[..., :x.shape[-1] // 2])
    return torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1).detach()


def complex_normalize(x: torch.Tensor) -> torch.Tensor:
    r"""Normalize the length of relation vectors, if the forward constraint has not been applied yet.

    The `modulus of complex number <https://en.wikipedia.org/wiki/Absolute_value#Complex_numbers>`_ is given as:

    .. math::

        |a + ib| = \sqrt{a^2 + b^2}

    $l_2$ norm of complex vector $x \in \mathbb{C}^d$:

    .. math::
        \|x\|^2 = \sum_{i=1}^d |x_i|^2
                 = \sum_{i=1}^d \left(\operatorname{Re}(x_i)^2 + \operatorname{Im}(x_i)^2\right)
                 = \left(\sum_{i=1}^d \operatorname{Re}(x_i)^2) + (\sum_{i=1}^d \operatorname{Im}(x_i)^2\right)
                 = \|\operatorname{Re}(x)\|^2 + \|\operatorname{Im}(x)\|^2
                 = \| [\operatorname{Re}(x); \operatorname{Im}(x)] \|^2
    """
    y = x.data.view(x.shape[0], -1, 2)
    y = functional.normalize(y, p=2, dim=-1)
    x.data = y.view(*x.shape)
    return x


class RotatE(SingleVectorEmbeddingModel):
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
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        # TODO: regularization
        super().__init__(
            triples_factory=triples_factory,
            interaction=RotatEInteraction(),
            embedding_dim=2 * embedding_dim,
            loss=loss,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            embedding_specification=EmbeddingSpecification(
                initializer=xavier_uniform_,
            ),
            relation_embedding_specification=EmbeddingSpecification(
                initializer=init_phases,
                constrainer=complex_normalize,
            ),
        )
