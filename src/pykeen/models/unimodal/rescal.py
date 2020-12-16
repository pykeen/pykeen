# -*- coding: utf-8 -*-

"""Implementation of RESCAL."""

from typing import Any, ClassVar, Mapping, Optional, Type

from ..base import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss
from ...nn import EmbeddingSpecification
from ...nn.modules import RESCALInteraction
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'RESCAL',
]


class RESCAL(ERModel):
    r"""An implementation of RESCAL from [nickel2011]_.

    This model represents relations as matrices and models interactions between latent features.

    RESCAL is a bilinear model that models entities as vectors and relations as matrices.
    The relation matrices $\textbf{W}_{r} \in \mathbb{R}^{d \times d}$ contain weights $w_{i,j}$ that
    capture the amount of interaction between the $i$-th latent factor of $\textbf{e}_h \in \mathbb{R}^{d}$ and the
    $j$-th latent factor of $\textbf{e}_t \in \mathbb{R}^{d}$.

    Thus, the plausibility score of $(h,r,t) \in \mathbb{K}$ is given by:

    .. math::

        f(h,r,t) = \textbf{e}_h^{T} \textbf{W}_{r} \textbf{e}_t = \sum_{i=1}^{d}\sum_{j=1}^{d} w_{ij}^{(r)}
        (\textbf{e}_h)_{i} (\textbf{e}_t)_{j}
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The regularizer used by [nickel2011]_ for for RESCAL
    #: According to https://github.com/mnick/rescal.py/blob/master/examples/kinships.py
    #: a normalized weight of 10 is used.
    regularizer_default: ClassVar[Type[Regularizer]] = LpRegularizer
    #: The LP settings used by [nickel2011]_ for for RESCAL
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=10,
        p=2.,
        normalize=True,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        loss: Optional[Loss] = None,
        regularizer: Optional[Regularizer] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        r"""Initialize RESCAL.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.

        .. seealso::

            - OpenKE `implementation of RESCAL <https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py>`_
        """
        if regularizer is None:
            regularizer = self._instantiate_default_regularizer()
        super().__init__(
            triples_factory=triples_factory,
            interaction=RESCALInteraction(),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                regularizer=regularizer,
            ),
            relation_representations=EmbeddingSpecification(
                shape=(embedding_dim, embedding_dim),
                regularizer=regularizer,
            ),
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
