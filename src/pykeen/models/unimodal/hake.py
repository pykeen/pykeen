# -*- coding: utf-8 -*-

"""Implementation of HAKE."""

from typing import Any, ClassVar, Mapping, Tuple

from torch import FloatTensor

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import EmbeddingSpecification
from ...nn.init import xavier_uniform_
from ...nn.modules import HAKEInteraction, ModEInteraction
from ...typing import Hint, Initializer

__all__ = [
    'HAKE',
    'ModE',
]


class HAKE(ERModel[Tuple[FloatTensor, FloatTensor], Tuple[FloatTensor, FloatTensor], Tuple[FloatTensor, FloatTensor]]):
    r"""An implementation of HAKE from [zhang2020]_.

    ---
    citation:
        author: Zhang
        year: 2020
        link: https://arxiv.org/abs/1911.09419
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,  # TODO check
        modulus_weight: float = 1.0,
        phase_weight: float = 0.5,
        entity_initializer: Hint[Initializer] = xavier_uniform_,  # TODO check
        relation_initializer: Hint[Initializer] = xavier_uniform_,  # TODO check
        **kwargs,
    ) -> None:
        r"""Initialize HAKE via the :class:`pykeen.nn.modules.HAKEInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$. Defaults to 50.
        :param modulus_weight:
            The initial weight for the modulus term.
        :param phase_weight:
            The initial weight for the phase term.
        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`
        :param relation_initializer: Relation initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=HAKEInteraction,
            interaction_kwargs=dict(
                modulus_weight=modulus_weight,
                phase_weight=phase_weight,
            ),
            entity_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=entity_initializer,
                ),
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=entity_initializer,
                ),
            ],
            relation_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                ),
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                ),
            ],
            **kwargs,
        )


class ModE(ERModel[FloatTensor, FloatTensor, FloatTensor]):
    r"""An implementation of ModE from [zhang2020]_.

    ---
    citation:
        author: Zhang
        year: 2020
        link: https://arxiv.org/abs/1911.09419
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,  # TODO check
        entity_initializer: Hint[Initializer] = xavier_uniform_,  # TODO check
        relation_initializer: Hint[Initializer] = xavier_uniform_,  # TODO check
        **kwargs,
    ) -> None:
        r"""Initialize ModE via the :class:`pykeen.nn.modules.ModEInteraction` interaction.

        :param embedding_dim: The entity embedding dimension $d$. Defaults to 50.
        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`
        :param relation_initializer: Relation initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`
        :param kwargs: Remaining keyword arguments passed through to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=ModEInteraction,
            entity_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=entity_initializer,
                ),
            ],
            relation_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                ),
            ],
            **kwargs,
        )
