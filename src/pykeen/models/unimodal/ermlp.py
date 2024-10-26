"""Implementation of ERMLP."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

from class_resolver import HintOrType, OptionalKwargs, ResolverKey, update_docstring_with_resolver_keys
from torch import nn

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import ERMLPInteraction
from ...typing import Hint, Initializer

__all__ = [
    "ERMLP",
]


class ERMLP(ERModel):
    r"""An implementation of ERMLP from [dong2014]_.

    This model represents both entities and relations as $d$-dimensional vectors stored in an
    :class:`~pykeen.nn.representation.Embedding` matrix.
    The representations are then passed to the :class:`~pykeen.nn.modules.ERMLPInteraction` function to obtain
    scores.

    ---
    name: ER-MLP
    citation:
        author: Dong
        year: 2014
        link: https://dl.acm.org/citation.cfm?id=2623623
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    @update_docstring_with_resolver_keys(
        ResolverKey(name="activation", resolver="class_resolver.contrib.torch.activation_resolver")
    )
    def __init__(
        self,
        *,
        embedding_dim: int = 64,
        hidden_dim: Optional[int] = None,
        activation: HintOrType[nn.Module] = nn.ReLU,
        activation_kwargs: OptionalKwargs = None,
        entity_initializer: Hint[Initializer] = nn.init.uniform_,
        relation_initializer: Hint[Initializer] = nn.init.uniform_,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            The embedding vector dimension for entities and relations.
        :param hidden_dim:
            The hidden dimension of the MLP. Defaults to `embedding_dim`.
        :param activation:
            The activation function or a hint thereof.
        :param activation_kwargs:
            Additional keyword-based parameters passed to the activation's constructor, if the activation is not
            pre-instantiated.
        :param entity_initializer:
            the method to initialize the entity embeddings
        :param relation_initializer:
            the method to initialize the entity embeddings
        :param kwargs:
            additional keyword-based parameters passed to :class:`pykeen.models.ERModel`
        """
        super().__init__(
            interaction=ERMLPInteraction,
            interaction_kwargs=dict(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                activation_kwargs=activation_kwargs,
            ),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
        )
