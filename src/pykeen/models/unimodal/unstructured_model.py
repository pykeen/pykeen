"""Implementation of UM."""

from collections.abc import Mapping
from typing import Any, ClassVar

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.init import xavier_normal_
from ...nn.modules import UMInteraction
from ...typing import FloatTensor, Hint, Initializer

__all__ = [
    "UM",
]


class UM(ERModel[FloatTensor, tuple[()], FloatTensor]):
    r"""An implementation of the Unstructured Model (UM) published by [bordes2014]_.

    This model represents entities as $d$-dimensional vectors stored in :class:`~pykeen.nn.representation.Embedding`.
    It does not have any relation representations. The :class:`~pykeen.nn.modules.UMInteraction` is used to
    calculate scores.

    ---
    name: Unstructured Model
    citation:
        author: Bordes
        year: 2014
        link: https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 1,
        power_norm: bool = False,
        entity_initializer: Hint[Initializer] = xavier_normal_,
        **kwargs,
    ) -> None:
        r"""Initialize UM.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.

        :param scoring_fct_norm:
            The norm used with :func:`torch.linalg.vector_norm`. Typically is 1 or 2.
        :param power_norm:
            Whether to use the p-th power of the $L_p$ norm. It has the advantage of being differentiable around 0,
            and numerically more stable.

        :param entity_initializer: The initializer for the entity embeddings.
            Defaults to :func:`pykeen.nn.init.xavier_normal`.

        :param kwargs: Remaining keyword arguments passed through to :class:`~pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=UMInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm, power_norm=power_norm),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations=[],
            **kwargs,
        )
