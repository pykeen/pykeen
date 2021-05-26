# -*- coding: utf-8 -*-

"""Implementation of CrossE."""

from typing import Any, Mapping, Optional, Tuple

from torch import FloatTensor, nn

from class_resolver import HintOrType
from ..nbase import ERModel
from ...nn.emb import EmbeddingSpecification
from ...nn.modules import CrossEInteraction

__all__ = [
    'CrossE',
]


class CrossE(ERModel[FloatTensor, Tuple[FloatTensor, FloatTensor], FloatTensor]):
    r"""An implementation of CrossE from [zhang2019b]_.

    ---
    citation:
        author: Zhang
        year: 2019
        link: https://arxiv.org/abs/1903.04750
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        combination_activation: HintOrType[nn.Module] = nn.Tanh,
        combination_activation_kwargs: Optional[Mapping[str, Any]] = None,
        combination_dropout: Optional[float] = 0.5,
        **kwargs,
    ) -> None:
        r"""Initialize CrossE via the :class:`pykeen.nn.modules.CrossEInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$. Defaults to 200. Is usually $d \in [50, 300]$.
        :param combination_activation:
            The combination activation function.
        :param combination_activation_kwargs:
            Additional keyword-based arguments passed to the constructor of the combination activation function (if
            not already instantiated).
        :param combination_dropout:
            An optional dropout applied to the combination.
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=CrossEInteraction,
            interaction_kwargs=dict(
                combination_activation=combination_activation,
                combination_activation_kwargs=combination_activation_kwargs,
                combination_dropout=combination_dropout,

            ),
            entity_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                ),
            ],
            relation_representations=[
                # Regular relation embeddings
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                ),
                # The relation-specific interaction vector
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                ),
            ],
            **kwargs,
        )
