"""Implementation of TransF."""

from collections.abc import Mapping
from typing import Any, ClassVar

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.modules import TransFInteraction
from ...typing import FloatTensor, Hint, Initializer, Normalizer

__all__ = [
    "TransF",
]


class TransF(ERModel[FloatTensor, FloatTensor, FloatTensor]):
    r"""An implementation of TransF from [feng2016]_.

    This model represents both entities and relations as $d$-dimensional vectors stored in an
    :class:`~pykeen.nn.representation.Embedding` matrix. The representations are then passed
    to the :class:`~pykeen.nn.modules.TransFInteraction` function to obtain scores.

    ---
    citation:
        author: Feng
        year: 2016
        link: https://www.aaai.org/ocs/index.php/KR/KR16/paper/view/12887
        arxiv: 1505.05253
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        embedding_dim: int = 128,
        entity_initializer: Hint[Initializer] = None,
        entity_initializer_kwargs: Mapping[str, Any] | None = None,
        entity_normalizer: Hint[Normalizer] = None,
        entity_normalizer_kwargs: Mapping[str, Any] | None = None,
        relation_initializer: Hint[Initializer] = None,
        relation_initializer_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        r"""Initialize the model.

        :param embedding_dim: The entity embedding dimension $d$.
        :param entity_initializer: Entity initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer
        :param entity_normalizer: Entity normalizer function. Defaults to :func:`torch.nn.functional.normalize`
        :param entity_normalizer_kwargs: Keyword arguments to be used when calling the entity normalizer
        :param relation_initializer: Relation initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=TransFInteraction,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                initializer_kwargs=entity_initializer_kwargs,
                normalizer=entity_normalizer,
                normalizer_kwargs=entity_normalizer_kwargs,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                initializer_kwargs=relation_initializer_kwargs,
            ),
            **kwargs,
        )
