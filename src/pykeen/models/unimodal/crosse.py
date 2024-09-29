"""Implementation of CrossE."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

from class_resolver import HintOrType, ResolverKey, update_docstring_with_resolver_keys
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

    CrossE represents each entity by a $d$-dimensional vector.
    Relations are represented by two $d$-dimensional vectors, one of which is a regular embedding vector,
    while the other is relation-specific interaction vector.
    All are stored in :class:`~pykeen.nn.representation.Embedding`.
    On top of that, :class:`~pykeen.nn.modules.CrossEInteraction` is used to get the scores.

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

    @update_docstring_with_resolver_keys(
        ResolverKey("combination_activation", "class_resolver.contrib.torch.activation_resolver")
    )
    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        combination_activation: HintOrType[nn.Module] = nn.Tanh,
        combination_activation_kwargs: Optional[Mapping[str, Any]] = None,
        combination_dropout: Optional[float] = 0.5,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = xavier_uniform_,
        relation_interaction_initializer: Hint[Initializer] = xavier_uniform_,
        **kwargs,
    ) -> None:
        r"""Initialize the model.

        :param embedding_dim:
            The entity and relation embedding dimension $d$. Defaults to 50.
        :param combination_activation:
            The combination activation function.
        :param combination_activation_kwargs:
            Additional keyword-based arguments passed to the constructor of the combination activation function (if
            not already instantiated).
        :param combination_dropout:
            An optional dropout applied after the combination and before the dot product similarity.
        :param entity_initializer:
            Entity initializer function.
        :param relation_initializer:
            Relation embedding initializer function.
        :param relation_interaction_initializer:
            Relation interaction vector initializer function.
        :param kwargs:
            Remaining keyword arguments passed through to :class:`~pykeen.models.ERModel`.
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
