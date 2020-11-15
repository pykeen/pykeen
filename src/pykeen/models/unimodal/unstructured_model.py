# -*- coding: utf-8 -*-

"""Implementation of UM."""

from typing import Optional

from .. import Model
from ...losses import Loss
from ...nn import Embedding
from ...nn.init import xavier_normal_
from ...nn.modules import UnstructuredModelInteraction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'UnstructuredModel',
]


class UnstructuredModel(Model):
    r"""An implementation of the Unstructured Model (UM) published by [bordes2014]_.

    UM computes the distance between head and tail entities then applies the $l_p$ norm.

    .. math::

        f(h, r, t) = - \|\textbf{e}_h  - \textbf{e}_t\|_p^2

    A small distance between the embeddings for the head and tail entity indicates a plausible triple. It is
    appropriate for networks with a single relationship type that is undirected.

    .. warning::

        In UM, neither the relations nor the directionality are considered, so it can't distinguish between them.
        However, it may serve as a baseline for comparison against relation-aware models.
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        scoring_fct_norm: int = 1,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize UM.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The $l_p$ norm. Usually 1 for UM.
        """
        self.embedding_dim = embedding_dim
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            interaction_function=UnstructuredModelInteraction(p=scoring_fct_norm),
            entity_representations=Embedding(
                num_embeddings=triples_factory.num_entities,
                embedding_dim=embedding_dim,
                initializer=xavier_normal_,
            ),
        )
