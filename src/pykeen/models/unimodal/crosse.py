"""Implementation of CrossE."""

from collections.abc import Mapping
from typing import Any, ClassVar

from class_resolver import HintOrType
from torch import FloatTensor, nn

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.init import xavier_uniform_
from ...nn.modules import CrossEInteraction
from ...typing import Hint, Initializer

__all__ = [
    "CrossE",
]


class CrossE(ERModel[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    r"""An implementation of CrossE from [zhang2019b]_.

    ---
    citation:
        author: Zhang
        year: 2019
        link: https://arxiv.org/abs/1903.04750
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        combination_activation: HintOrType[nn.Module] = nn.Tanh,
        combination_activation_kwargs: Mapping[str, Any] | None = None,
        combination_dropout: float | None = 0.5,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = xavier_uniform_,
        relation_interaction_initializer: Hint[Initializer] = xavier_uniform_,
        **kwargs,
    ) -> None:
        r"""Initialize CrossE via the :class:`pykeen.nn.modules.CrossEInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$. Defaults to 50.
        :param combination_activation:
            The combination activation function.
        :param combination_activation_kwargs:
            Additional keyword-based arguments passed to the constructor of the combination activation function (if
            not already instantiated).
        :param combination_dropout:
            An optional dropout applied to the combination.
        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`
        :param relation_initializer: Relation initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`
        :param relation_interaction_initializer: Relation interaction vector initializer function. Defaults to
            :func:`pykeen.nn.init.xavier_uniform_`
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=CrossEInteraction,
            interaction_kwargs=dict(
                combination_activation=combination_activation,
                combination_activation_kwargs=combination_activation_kwargs,
                combination_dropout=combination_dropout,
                embedding_dim=embedding_dim,
            ),
            entity_representations_kwargs=[
                dict(
                    shape=embedding_dim,
                    initializer=entity_initializer,
                ),
            ],
            relation_representations_kwargs=[
                # Regular relation embeddings
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                ),
                # The relation-specific interaction vector
                dict(
                    shape=embedding_dim,
                    initializer=relation_interaction_initializer,
                ),
            ],
            **kwargs,
        )
