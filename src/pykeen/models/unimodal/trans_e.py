"""TransE."""

from collections.abc import Mapping
from typing import Any, ClassVar

from class_resolver import Hint, HintOrType, OptionalKwargs
from torch.nn import functional

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import TransEInteraction
from ...nn.init import xavier_uniform_, xavier_uniform_norm_
from ...regularizers import Regularizer
from ...typing import Constrainer, FloatTensor, Initializer

__all__ = [
    "TransE",
]


class TransE(ERModel[FloatTensor, FloatTensor, FloatTensor]):
    r"""
    An implementation of TransE [bordes2013]_.

    This model represents both entities and relations as $d$-dimensional vectors stored in an
    :class:`~pykeen.nn.representation.Embedding` matrix. The representations are then passed
    to the :class:`~pykeen.nn.modules.TransEInteraction` function to obtain scores.
    ---
    citation:
        author: Bordes
        year: 2013
        link: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
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
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = None,
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        r"""Initialize TransE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.

        :param scoring_fct_norm:
            The norm used with :func:`torch.linalg.vector_norm`. Typically is 1 or 2.
        :param power_norm:
            Whether to use the p-th power of the $L_p$ norm. It has the advantage of being differentiable around 0,
            and numerically more stable.

        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`.
        :param entity_constrainer: Entity constrainer function. Defaults to :func:`torch.nn.functional.normalize`.

        :param relation_initializer:
            Relation initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_norm_`.
        :param relation_constrainer: Relation constrainer function. Defaults to none.

        :param regularizer:
            a regularizer, or a hint thereof. Used for both, entity and relation representations;
            directly use :class:`~pykeen.models.ERModel` if you need more flexibility
        :param regularizer_kwargs:
            keyword-based parameters for the regularizer

        :param kwargs:
            Remaining keyword arguments to forward to :meth:`~pykeen.models.ERModel.__init__`

        .. seealso::

           - OpenKE `implementation of TransE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
           - :class:`~pykeen.nn.modules.NormBasedInteraction` for a description of the parameters
                ``scoring_fct_norm`` and ``power_norm``.
        """
        super().__init__(
            interaction=TransEInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm, power_norm=power_norm),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            **kwargs,
        )
