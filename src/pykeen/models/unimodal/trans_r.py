# -*- coding: utf-8 -*-

"""Implementation of TransR."""
import logging
from typing import Optional

from torch.nn import functional

from .. import Model
from ...losses import Loss
from ...nn import Embedding
from ...nn.emb import EmbeddingSpecification
from ...nn.init import xavier_uniform_
from ...nn.modules import TransRInteractionFunction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import clamp_norm, compose

__all__ = [
    'TransR',
]


class TransR(Model):
    r"""An implementation of TransR from [lin2015]_.

    TransR is an extension of :class:`pykeen.models.TransH` that explicitly considers entities and relations as
    different objects and therefore represents them in different vector spaces.

    For a triple $(h,r,t) \in \mathbb{K}$, the entity embeddings, $\textbf{e}_h, \textbf{e}_t \in \mathbb{R}^d$,
    are first projected into the relation space by means of a relation-specific projection matrix
    $\textbf{M}_{r} \in \mathbb{R}^{k \times d}$. With relation embedding $\textbf{r}_r \in \mathbb{R}^k$, the
    interaction model is defined similarly to TransE with:

    .. math::

        f(h,r,t) = -\|\textbf{M}_{r}\textbf{e}_h + \textbf{r}_r - \textbf{M}_{r}\textbf{e}_t\|_{p}^2

    The following constraints are applied:

     * $\|\textbf{e}_h\|_2 \leq 1$
     * $\|\textbf{r}_r\|_2 \leq 1$
     * $\|\textbf{e}_t\|_2 \leq 1$
     * $\|\textbf{M}_{r}\textbf{e}_h\|_2 \leq 1$
     * $\|\textbf{M}_{r}\textbf{e}_t\|_2 \leq 1$

    .. seealso::

       - OpenKE `TensorFlow implementation of TransR
         <https://github.com/thunlp/OpenKE/blob/master/models/TransR.py>`_
       - OpenKE `PyTorch implementation of TransR
         <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransR.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=20, high=300, q=50),
        relation_dim=dict(type=int, low=20, high=300, q=50),
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        relation_dim: int = 30,
        scoring_fct_norm: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            entity_representations=Embedding.from_specification(
                num_embeddings=triples_factory.num_entities,
                shape=embedding_dim,
                specification=EmbeddingSpecification(
                    initializer=xavier_uniform_,
                    constrainer=clamp_norm,
                    constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
                )
            ),
            relation_representations=[
                Embedding.from_specification(
                    triples_factory.num_relations,
                    shape=relation_dim,
                    specification=EmbeddingSpecification(
                        initializer=compose(
                            xavier_uniform_,
                            functional.normalize,
                        ),
                        constrainer=clamp_norm,
                        constrainer_kwargs=dict(maxnorm=1., p=2, dim=-1),
                    ),
                ),
                # Relation projections
                Embedding.from_specification(
                    triples_factory.num_relations,
                    shape=(relation_dim, embedding_dim),
                    specification=EmbeddingSpecification(
                        initializer=xavier_uniform_,
                    ),
                )
            ],
            interaction_function=TransRInteractionFunction(
                p=scoring_fct_norm,
            ),
        )
        logging.warning("Initialize from TransE")
