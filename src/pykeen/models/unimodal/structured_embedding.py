"""Implementation of structured model (SE)."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

from class_resolver import Hint
from torch.nn import functional

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.init import xavier_uniform_, xavier_uniform_norm_
from ...nn.modules import SEInteraction
from ...typing import Constrainer, FloatTensor, Initializer

__all__ = [
    "SE",
]


class SE(ERModel[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    r"""An implementation of the Structured Embedding (SE) published by [bordes2011]_.

    This model represents entities as $d$-dimensional vectors, and relations by two projection matrices
    $\textbf{M}_{r}^{h}, \textbf{M}_{r}^{t} \in \mathbb{R}^{d \times d}$ for the head and tail role respectively.
    They are stored in an :class:`~pykeen.nn.representation.Embedding` matrix. The representations are then passed
    to the :class:`~pykeen.nn.modules.SEInteraction` function to obtain scores.

    ---
    name: Structured Embedding
    citation:
        author: Bordes
        year: 2011
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898
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
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        entity_constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        **kwargs,
    ) -> None:
        r"""Initialize SE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.

        :param scoring_fct_norm:
            The norm used with :func:`torch.linalg.vector_norm`. Typically is 1 or 2.
        :param power_norm:
            Whether to use the p-th power of the $L_p$ norm. It has the advantage of being differentiable around 0,
            and numerically more stable.

        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`.
        :param entity_constrainer: Entity constrainer function. Defaults to :func:`torch.nn.functional.normalize`.
        :param entity_constrainer_kwargs: Keyword arguments to be used when calling the entity constrainer.

        :param relation_initializer: Relation initializer function. Defaults to
            :func:`pykeen.nn.init.xavier_uniform_norm_`

        :param kwargs:
            Remaining keyword arguments to forward to :class:`~pykeen.models.ERModel`
        """
        super().__init__(
            interaction=SEInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm, power_norm=power_norm),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                constrainer_kwargs=entity_constrainer_kwargs,
            ),
            relation_representations_kwargs=[
                dict(
                    shape=(embedding_dim, embedding_dim),
                    initializer=relation_initializer,
                ),
                dict(
                    shape=(embedding_dim, embedding_dim),
                    initializer=relation_initializer,
                ),
            ],
            **kwargs,
        )
