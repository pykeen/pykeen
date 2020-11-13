# -*- coding: utf-8 -*-

"""Implementation of the HolE model."""

from typing import Optional

from ..base import SingleVectorEmbeddingModel
from ...losses import Loss
from ...nn.init import xavier_uniform_
from ...nn.modules import HolEInteractionFunction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import clamp_norm

__all__ = [
    'HolE',
]


class HolE(SingleVectorEmbeddingModel):
    r"""An implementation of HolE [nickel2016]_.

    Holographic embeddings (HolE) make use of the circular correlation operator to compute interactions between
    latent features of entities and relations:

    .. math::

        f(h,r,t) = \sigma(\textbf{r}^{T}(\textbf{h} \star \textbf{t}))

    where the circular correlation $\star: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^d$ is defined as:

    .. math::

        [\textbf{a} \star \textbf{b}]_i = \sum_{k=0}^{d-1} \textbf{a}_{k} * \textbf{b}_{(i+k)\ mod \ d}

    By using the correlation operator each component $[\textbf{h} \star \textbf{t}]_i$ represents a sum over a
    fixed partition over pairwise interactions. This enables the model to put semantic similar interactions into the
    same partition and share weights through $\textbf{r}$. Similarly irrelevant interactions of features could also
    be placed into the same partition which could be assigned a small weight in $\textbf{r}$.

    .. seealso::

       - `author's implementation of HolE <https://github.com/mnick/holographic-embeddings>`_
       - `scikit-kge implementation of HolE <https://github.com/mnick/scikit-kge>`_
       - OpenKE `implementation of HolE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
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
        """Initialize the model."""
        interaction_function = HolEInteractionFunction()

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            loss=loss,
            interaction_function=interaction_function,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            # Initialisation, cf. https://github.com/mnick/scikit-kge/blob/master/skge/param.py#L18-L27
            entity_initializer=xavier_uniform_,
            relation_initializer=xavier_uniform_,
            entity_constrainer=clamp_norm,
            entity_constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
        )
