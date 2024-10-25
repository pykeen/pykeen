"""Implementation of the QuatE model."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

import torch

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEWithLogitsLoss, Loss
from ...nn import quaternion
from ...nn.init import init_quaternions
from ...nn.modules import QuatEInteraction
from ...regularizers import LpRegularizer, Regularizer
from ...typing import Constrainer, Hint, Initializer
from ...utils import get_expected_norm

__all__ = [
    "QuatE",
]


class QuatE(ERModel):
    r"""An implementation of QuatE from [zhang2019]_.

    QuatE uses hypercomplex valued representations for the
    entities and relations. Entities and relations are represented as vectors
    $\textbf{e}_i, \textbf{r}_i \in \mathbb{H}^d$, and the plausibility score is computed using the
    quaternion inner product.

    The representations are stored in an :class:`~pykeen.nn.representation.Embedding`.
    Scores are calculated with :class:`~pykeen.nn.modules.QuatEInteraction`.

    .. seealso ::

        Official implementation: https://github.com/cheungdaven/QuatE/blob/master/models/QuatE.py
    ---
    citation:
        author: Zhang
        year: 2019
        arxiv: 1904.10281
        link: https://arxiv.org/abs/1904.10281
        github: cheungdaven/quate
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[type[Loss]] = BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = dict(reduction="mean")
    #: The LP settings used by [zhang2019]_ for QuatE.
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.3 / get_expected_norm(p=2, d=100),
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 100,
        entity_initializer: Hint[Initializer] = init_quaternions,
        entity_regularizer: Hint[Regularizer] = LpRegularizer,
        entity_regularizer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = init_quaternions,
        relation_regularizer: Hint[Regularizer] = LpRegularizer,
        relation_regularizer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_normalizer: Hint[Constrainer] = quaternion.normalize,
        **kwargs,
    ) -> None:
        """Initialize QuatE.

        .. note ::
            The default parameters correspond to the first setting for FB15k-237 described from [zhang2019]_.

        :param embedding_dim:
            The embedding dimensionality of the entity embeddings.

            .. note ::
                The number of parameter per entity is `4 * embedding_dim`, since quaternion are used.

        :param entity_initializer:
            The initializer to use for the entity embeddings.
        :param entity_regularizer:
            The regularizer to use for the entity embeddings.
        :param entity_regularizer_kwargs:
            The keyword arguments passed to the entity regularizer. Defaults to
            :data:`QuatE.regularizer_default_kwargs` if not specified.
        :param relation_initializer:
            The initializer to use for the relation embeddings.
        :param relation_regularizer:
            The regularizer to use for the relation embeddings.
        :param relation_regularizer_kwargs:
            The keyword arguments passed to the relation regularizer. Defaults to
            :data:`QuatE.regularizer_default_kwargs` if not specified.
        :param relation_normalizer:
            The normalizer to use for the relation embeddings.
        :param kwargs:
            Additional keyword based arguments passed to :class:`pykeen.models.ERModel`. Must not contain
            "interaction", "entity_representations", or "relation_representations".
        """
        super().__init__(
            interaction=QuatEInteraction,
            entity_representations_kwargs=dict(
                shape=(embedding_dim, 4),  # quaternions
                initializer=entity_initializer,
                dtype=torch.float,
                regularizer=entity_regularizer,
                regularizer_kwargs=entity_regularizer_kwargs or self.regularizer_default_kwargs,
            ),
            relation_representations_kwargs=dict(
                shape=(embedding_dim, 4),  # quaternions
                initializer=relation_initializer,
                normalizer=relation_normalizer,
                dtype=torch.float,
                regularizer=relation_regularizer,
                regularizer_kwargs=relation_regularizer_kwargs or self.regularizer_default_kwargs,
            ),
            **kwargs,
        )
