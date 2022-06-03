# -*- coding: utf-8 -*-

"""Implementation of RESCAL."""

from typing import Any, ClassVar, Mapping, Type

from class_resolver import HintOrType, OptionalKwargs
from torch.nn.init import uniform_

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import RESCALInteraction
from ...regularizers import LpRegularizer, Regularizer
from ...typing import Hint, Initializer

__all__ = [
    "RESCAL",
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
    ---
    citation:
        author: Nickel
        year: 2011
        link: http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf
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
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        entity_initializer: Hint[Initializer] = uniform_,
        relation_initializer: Hint[Initializer] = uniform_,
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        r"""Initialize RESCAL.

        :param embedding_dim:
            the entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param entity_initializer:
            entity initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param relation_initializer:
            relation initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param regularizer:
            the regularizer. Default to :attr:`RESCAL.defaul_regularizer`
        :param regularizer_kwargs:
            additional keyword-based parameters for the regularizer
        :param kwargs:
            remaining keyword arguments to forward to :meth:`pykeen.models.ERModel.__init__`

        .. seealso::

            - OpenKE `implementation of RESCAL <https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py>`_
        """
        regularizer = self._instantiate_regularizer(regularizer=regularizer, regularizer_kwargs=regularizer_kwargs)
        super().__init__(
            interaction=RESCALInteraction,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                regularizer=regularizer,
            ),
            relation_representations_kwargs=dict(
                shape=(embedding_dim, embedding_dim),  # d x d matrices
                initializer=relation_initializer,
                regularizer=regularizer,
            ),
            **kwargs,
        )
