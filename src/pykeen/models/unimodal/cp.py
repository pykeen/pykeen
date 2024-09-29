"""Implementation of CP model."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.modules import CPInteraction
from ...typing import Hint, Initializer, Normalizer

__all__ = [
    "CP",
]


class CP(ERModel):
    r"""An implementation of CP as described in [lacroix2018]_ based on [hitchcock1927]_.

    It has separate entity representations for the head and tail role, both a $r \times d$-dimensional matrices.
    Relations are also represented by a $r \times d$-dimensional matrix.
    All three components can be stored as :class:`~pykeen.nn.representation.Embedding`.

    On top of these, :class:`~pykeen.nn.modules.CPInteraction` is applied to obtain scores.

    ---
    name: Canonical Tensor Decomposition
    citation:
        author: Lacroix
        year: 2018
        arxiv: 1806.07297
        link: https://arxiv.org/abs/1806.07297
        github: facebookresearch/kbc
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE, rank=dict(type=int, low=2, high=2048, log=True)
    )

    def __init__(
        self,
        embedding_dim: int = 64,
        rank: int = 64,
        entity_initializer: Hint[Initializer] = None,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_normalizer: Hint[Normalizer] = None,
        entity_normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = None,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize the model.

        :param embedding_dim: The entity embedding dimension $d$.
        :param rank: The tensor decomposition rank $k$.
        :param entity_initializer: Entity initializer function. Defaults to None
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer
        :param entity_normalizer: Entity normalizer function. Defaults to None
        :param entity_normalizer_kwargs: Keyword arguments to be used when calling the entity normalizer
        :param relation_initializer: Relation initializer function. Defaults to None
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer
        :param kwargs: Remaining keyword arguments passed through to :class:`~pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=CPInteraction,
            entity_representations_kwargs=[
                # head representation
                dict(
                    shape=(rank, embedding_dim),
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs,
                    normalizer=entity_normalizer,
                    normalizer_kwargs=entity_normalizer_kwargs,
                ),
                # tail representation
                dict(
                    shape=(rank, embedding_dim),
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs,
                    normalizer=entity_normalizer,
                    normalizer_kwargs=entity_normalizer_kwargs,
                ),
            ],
            relation_representations_kwargs=dict(
                shape=(rank, embedding_dim),
                initializer=relation_initializer,
                initializer_kwargs=relation_initializer_kwargs,
            ),
            **kwargs,
        )
