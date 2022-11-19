# -*- coding: utf-8 -*-

"""Implementation of the RotatE model."""

from typing import Any, ClassVar, Mapping

import torch
from class_resolver import HintOrType, OptionalKwargs

from ..nbase import ERModel
from ...nn.init import init_phases, xavier_uniform_
from ...nn.modules import RotatEInteraction
from ...regularizers import Regularizer
from ...typing import Constrainer, Hint, Initializer
from ...utils import complex_normalize

__all__ = [
    "RotatE",
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
    ---
    citation:
        author: Sun
        year: 2019
        link: https://arxiv.org/abs/1902.10197v1
        github: DeepGraphLearning/KnowledgeGraphEmbedding
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=dict(type=int, low=32, high=1024, q=16),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = init_phases,
        relation_constrainer: Hint[Constrainer] = complex_normalize,
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            the embedding dimension
        :param entity_initializer:
            the entity representation initializer
        :param relation_initializer:
            the relation representation initializer
        :param relation_constrainer:
            the relation representation constrainer
        :param regularizer:
            the regularizer
        :param regularizer_kwargs:
            additional keyword-based parameters passed to the regularizer
        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        super().__init__(
            interaction=RotatEInteraction,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
                dtype=torch.cfloat,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
                dtype=torch.cfloat,
            ),
            **kwargs,
        )
