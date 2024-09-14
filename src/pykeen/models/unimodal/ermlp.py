"""Implementation of ERMLP."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

from torch.nn.init import uniform_

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

    def __init__(
        self,
        *,
        embedding_dim: int = 64,
        hidden_dim: Optional[int] = None,
        entity_initializer: Hint[Initializer] = uniform_,
        relation_initializer: Hint[Initializer] = uniform_,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        # input normalization
        if hidden_dim is None:
            hidden_dim = embedding_dim
        super().__init__(
            interaction=ERMLPInteraction,
            interaction_kwargs=dict(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
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
