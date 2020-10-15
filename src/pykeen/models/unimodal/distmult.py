# -*- coding: utf-8 -*-

"""Implementation of DistMult."""

from typing import Optional

import torch.autograd
from torch import nn
from torch.nn import functional

from ..base import InteractionFunction, SimpleVectorEntityRelationEmbeddingModel, normalize_for_einsum
from ...losses import Loss
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory

__all__ = [
    'DistMult',
    'DistMultInteractionFunction',
]


class DistMultInteractionFunction(InteractionFunction):
    """Interaction function of DistMult."""

    def forward(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        batch_size = max(h.shape[0], r.shape[0], t.shape[0])
        h_term, h = normalize_for_einsum(x=h, batch_size=batch_size, symbol='h')
        r_term, r = normalize_for_einsum(x=r, batch_size=batch_size, symbol='r')
        t_term, t = normalize_for_einsum(x=t, batch_size=batch_size, symbol='t')
        return torch.einsum(f'{h_term},{r_term},{t_term}->bhrt', h, r, t)


class DistMult(SimpleVectorEntityRelationEmbeddingModel):
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
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize DistMult.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        """
        super().__init__(
            triples_factory=triples_factory,
            interaction_function=DistMultInteractionFunction(),
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

    def _reset_parameters_(self):  # noqa: D102
        # xavier uniform, cf.
        # https://github.com/thunlp/OpenKE/blob/adeed2c0d2bef939807ed4f69c1ea4db35fd149b/models/DistMult.py#L16-L17
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        # Initialise relation embeddings to unit length
        functional.normalize(self.relation_embeddings.weight.data, out=self.relation_embeddings.weight.data)

    def post_parameter_update(self) -> None:  # noqa: D102
        # Make sure to call super first
        super().post_parameter_update()

        # Normalize embeddings of entities
        functional.normalize(self.entity_embeddings.weight.data, out=self.entity_embeddings.weight.data)
