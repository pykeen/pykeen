"""Implementation of TransD."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

from class_resolver import OptionalKwargs

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.init import xavier_normal_, xavier_uniform_, xavier_uniform_norm_
from ...nn.modules import TransDInteraction
from ...typing import Constrainer, FloatTensor, Hint, Initializer
from ...utils import clamp_norm

__all__ = [
    "TransD",
]


class TransD(
    ERModel[tuple[FloatTensor, FloatTensor], tuple[FloatTensor, FloatTensor], tuple[FloatTensor, FloatTensor]]
):
    r"""An implementation of TransD from [ji2015]_.

    This model represents both entities as pairs of $d$-dimensional vectors,
    and relations as pairs of $k$-dimensional vectors.
    Both, entity and relation embedding vectors are constrained to $\|\cdot\|_2 \leq 1$.
    They are stored in an :class:`~pykeen.nn.representation.Embedding` matrix.
    The representations are then passed to the :class:`~pykeen.nn.modules.TransDInteraction` function to obtain scores.

    .. seealso::

       - OpenKE `implementation of TransD <https://github.com/thunlp/OpenKE/blob/master/models/TransD.py>`_
    ---
    citation:
        author: Ji
        year: 2015
        link: http://www.aclweb.org/anthology/P15-1067
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        relation_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        relation_dim: Optional[int] = None,
        interaction_kwargs: OptionalKwargs = None,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            The (entity) embedding dimension.
        :param relation_dim:
            The relation embedding dimension. Defaults to ``embedding_dim``.
        :param interaction_kwargs:
            Additional keyword-based parameters passed to :class:`~pykeen.nn.modules.TransDInteraction`.

        :param entity_initializer:
            The entity representation initializer. Defaults to :func:`pykeen.nn.init.xavier_uniform_`.
        :param entity_constrainer:
            The entity representation constrainer. Defaults to :func:`pykeen.utils.clamp_norm`.

        :param relation_initializer:
            The relation representation initializer. Defaults to :func:`pykeen.nn.init.xavier_uniform_norm_`.
        :param relation_constrainer:
            The relation representation constrainer. Defaults to :func:`pykeen.utils.clamp_norm`.

        :param kwargs:
            Additional keyword-based parameters passed to :class:`~pykeen.models.ERModel`.
        """
        relation_dim = relation_dim or embedding_dim
        super().__init__(
            interaction=TransDInteraction,
            interaction_kwargs=interaction_kwargs,
            entity_representations_kwargs=[
                dict(
                    shape=embedding_dim,
                    initializer=entity_initializer,
                    constrainer=entity_constrainer,
                    constrainer_kwargs=dict(maxnorm=1.0, p=2, dim=-1),
                ),
                dict(
                    shape=embedding_dim,
                    initializer=xavier_normal_,
                ),
            ],
            relation_representations_kwargs=[
                dict(
                    shape=(relation_dim,),
                    initializer=relation_initializer,
                    constrainer=relation_constrainer,
                    constrainer_kwargs=dict(maxnorm=1.0, p=2, dim=-1),
                ),
                dict(
                    shape=(relation_dim,),
                    initializer=xavier_normal_,
                ),
            ],
            **kwargs,
        )
