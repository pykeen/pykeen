# -*- coding: utf-8 -*-

"""Implementation of TransR."""

from typing import Any, ClassVar, Mapping

import torch
import torch.autograd
import torch.nn.init

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import TransRInteraction
from ...nn.init import xavier_uniform_, xavier_uniform_norm_
from ...typing import Constrainer, Hint, Initializer
from ...utils import clamp_norm

__all__ = [
    "TransR",
]


class TransR(ERModel):
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
    ---
    citation:
        author: Lin
        year: 2015
        link: http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        relation_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        relation_dim: int = 30,
        scoring_fct_norm: int = 1,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        **kwargs,
    ) -> None:
        """Initialize the model."""
        # TODO: Initialize from TransE
        super().__init__(
            interaction=TransRInteraction,
            interaction_kwargs=dict(
                p=scoring_fct_norm,
            ),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                constrainer_kwargs=dict(maxnorm=1.0, p=2, dim=-1),
            ),
            relation_representations_kwargs=[
                # relation embedding
                dict(
                    shape=(relation_dim,),
                    initializer=relation_initializer,
                    constrainer=relation_constrainer,
                    constrainer_kwargs=dict(maxnorm=1.0, p=2, dim=-1),
                ),
                # relation projection
                dict(
                    shape=(embedding_dim, relation_dim),
                    initializer=torch.nn.init.xavier_uniform_,
                ),
            ],
            **kwargs,
        )
