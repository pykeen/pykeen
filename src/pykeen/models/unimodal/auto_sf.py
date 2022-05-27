# -*- coding: utf-8 -*-

"""A simple AutoSF-based model."""

from typing import Any, ClassVar, Mapping, Optional, Sequence, Tuple

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.modules import AutoSFInteraction
from ...typing import Sign

__all__ = [
    "AutoSF",
]

YAGO310_COEFFICIENTS: Sequence[Tuple[int, int, int, Sign]] = [
    # diagonal entries
    (0, 0, 0, 1),
    (1, 1, 1, 1),
    (2, 2, 2, 1),
    (3, 3, 3, 1),
    # off-diagonal
    (1, 2, 3, -1),
    (3, 1, 1, -1),
]


class AutoSF(ERModel):
    r"""An implementation of AutoSF from [zhang2020]_.

    ---
    name: AutoSF
    citation:
        author: Zhang
        year: 2020
        arxiv: 1904.11682
        link: https://arxiv.org/abs/1904.11682
        github: AutoML-Research/AutoSF
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        embedding_dim: int = 256,
        num_components: int = 4,
        coefficients: Sequence[Tuple[int, int, int, Sign]] = YAGO310_COEFFICIENTS,
        embedding_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize AutoSF via the :class:`pykeen.nn.modules.AutoSFInteraction` interaction.

        .. note::
            this variant uses `num_components` entity and relation representations with shared configuration.
            The coefficients should only be in $[0, num_components)$.

        :param embedding_dim: The entity embedding dimension $d$.
        :param num_components: the number of components.
        :param coefficients:
            the coefficients determining the structure. The coefficients describe which head/relation/tail
            component get combined with each other. While in theory, we can have up to `num_components**3`
            unique triples, usually, a smaller number is preferable to have some sparsity.
        :param embedding_kwargs: keyword arguments passed to the entity representation
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        embedding_kwargs = embedding_kwargs or {}
        super().__init__(
            interaction=AutoSFInteraction,
            interaction_kwargs=dict(coefficients=coefficients),
            entity_representations_kwargs=[
                dict(
                    shape=embedding_dim,
                    **embedding_kwargs,
                )
                for _ in range(num_components)
            ],
            relation_representations_kwargs=[
                dict(
                    shape=embedding_dim,
                    **embedding_kwargs,
                )
                for _ in range(num_components)
            ],
            **kwargs,
        )
