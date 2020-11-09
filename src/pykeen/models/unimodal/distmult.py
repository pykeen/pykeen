# -*- coding: utf-8 -*-

"""Implementation of DistMult."""

from typing import Optional

import torch
import torch.autograd
from torch import nn
from torch.nn import functional

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import compose

__all__ = [
    'DistMult',
]


class DistMult(EntityRelationEmbeddingModel):
    r"""An implementation of DistMult from [yang2014]_.

    This model simplifies RESCAL by restricting matrices representing relations as diagonal matrices.

    DistMult is a simplification of :class:`pykeen.models.RESCAL` where the relation matrices
    $\textbf{W}_{r} \in \mathbb{R}^{d \times d}$ are restricted to diagonal matrices:

    .. math::

        f(h,r,t) = \textbf{e}_h^{T} \textbf{W}_r \textbf{e}_t = \sum_{i=1}^{d}(\textbf{e}_h)_i \cdot
        diag(\textbf{W}_r)_i \cdot (\textbf{e}_t)_i

    Because of its restriction to diagonal matrices, DistMult is more computationally than RESCAL, but at the same
    time it is less expressive. For instance, it is not able to model anti-symmetric relations,
    since $f(h,r, t) = f(t,r,h)$. This can alternatively be formulated with relation vectors
    $\textbf{r}_r \in \mathbb{R}^d$ and the Hadamard operator and the $l_1$ norm.

    .. math::

        f(h,r,t) = \|\textbf{e}_h \odot \textbf{r}_r \odot \textbf{e}_t\|_1

    Note:
      - For FB15k, Yang *et al.* report 2 negatives per each positive.

    .. seealso::

       - OpenKE `implementation of DistMult <https://github.com/thunlp/OpenKE/blob/master/models/DistMult.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
    )
    #: The regularizer used by [yang2014]_ for DistMult
    #: In the paper, they use weight of 0.0001, mini-batch-size of 10, and dimensionality of vector 100
    #: Thus, when we use normalized regularization weight, the normalization factor is 10*sqrt(100) = 100, which is
    #: why the weight has to be increased by a factor of 100 to have the same configuration as in the paper.
    regularizer_default = LpRegularizer
    #: The LP settings used by [yang2014]_ for DistMult
    regularizer_default_kwargs = dict(
        weight=0.1,
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize DistMult.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            # xavier uniform, cf.
            # https://github.com/thunlp/OpenKE/blob/adeed2c0d2bef939807ed4f69c1ea4db35fd149b/models/DistMult.py#L16-L17
            entity_initializer=nn.init.xavier_uniform_,
            # Constrain entity embeddings to unit length
            entity_constrainer=functional.normalize,
            # relations are initialized to unit length (but not constraint)
            relation_initializer=compose(
                nn.init.xavier_uniform_,
                functional.normalize,
            ),
        )

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function for given embeddings.

        The embeddings have to be in a broadcastable shape.

        WARNING: Does not ensure forward constraints.

        :param h: shape: (..., e)
            Head embeddings.
        :param r: shape: (..., e)
            Relation embeddings.
        :param t: shape: (..., e)
            Tail embeddings.

        :return: shape: (...)
            The scores.
        """
        # Bilinear product
        # *: Elementwise multiplication
        return torch.sum(h * r * t, dim=-1)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hrt_batch[:, 0])
        r = self.relation_embeddings(hrt_batch[:, 1])
        t = self.entity_embeddings(hrt_batch[:, 2])

        # Compute score
        scores = self.interaction_function(h=h, r=r, t=t).view(-1, 1)

        # Only regularize relation embeddings
        self.regularize_if_necessary(r)

        return scores

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0]).view(-1, 1, self.embedding_dim)
        r = self.relation_embeddings(indices=hr_batch[:, 1]).view(-1, 1, self.embedding_dim)
        t = self.entity_embeddings(indices=None).view(1, -1, self.embedding_dim)

        # Rank against all entities
        scores = self.interaction_function(h=h, r=r, t=t)

        # Only regularize relation embeddings
        self.regularize_if_necessary(r)

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=None).view(1, -1, self.embedding_dim)
        r = self.relation_embeddings(indices=rt_batch[:, 0]).view(-1, 1, self.embedding_dim)
        t = self.entity_embeddings(indices=rt_batch[:, 1]).view(-1, 1, self.embedding_dim)

        # Rank against all entities
        scores = self.interaction_function(h=h, r=r, t=t)

        # Only regularize relation embeddings
        self.regularize_if_necessary(r)

        return scores
