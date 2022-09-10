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

    The AutoSF model combines one or more :class:`pykeen.nn.Embedding`s for entities and relations with a
    :class:`pykeen.nn.AutoSFInteraction` describing the interaction thereof.

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
        r"""Initialize AutoSF via the :class:`pykeen.nn.AutoSFInteraction` interaction.

        .. note::
            this variant uses `num_components` entity and relation representations with shared configuration.
            The coefficients should only be in $[0, num_components)$.

        :param embedding_dim:
            the entity embedding dimension $d$ for each block.
        :param num_components:
            the number of components/blocks.
        :param coefficients:
            the coefficients determining the block structure, cf. :class:`pykeen.nn.AutoSFInteraction`.
        :param embedding_kwargs:
            keyword arguments passed to the entity and relation representation
        :param kwargs:
            remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        embedding_kwargs = embedding_kwargs or {}
        super().__init__(
            interaction=AutoSFInteraction,
            interaction_kwargs=dict(num_blocks=num_components, coefficients=coefficients),
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
