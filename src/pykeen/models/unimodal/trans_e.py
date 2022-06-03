# -*- coding: utf-8 -*-

"""TransE."""

from typing import Any, ClassVar, Mapping

from class_resolver import Hint, HintOrType, OptionalKwargs
from torch.nn import functional

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import TransEInteraction
from ...nn.init import xavier_uniform_, xavier_uniform_norm_
from ...regularizers import Regularizer
from ...typing import Constrainer, Initializer

__all__ = [
    "TransE",
]


class TransE(ERModel):
    r"""An implementation of TransE [bordes2013]_.

    TransE models relations as a translation from head to tail entities in :math:`\textbf{e}`:

    .. math::

        \textbf{e}_h + \textbf{e}_r \approx \textbf{e}_t

    This equation is rearranged and the :math:`l_p` norm is applied to create the TransE interaction function.

    .. math::

        f(h, r, t) = - \|\textbf{e}_h + \textbf{e}_r - \textbf{e}_t\|_{p}

    While this formulation is computationally efficient, it inherently cannot model one-to-many, many-to-one, and
    many-to-many relationships. For triples :math:`(h,r,t_1), (h,r,t_2) \in \mathcal{K}` where :math:`t_1 \neq t_2`,
    the model adapts the embeddings in order to ensure :math:`\textbf{e}_h + \textbf{e}_r \approx \textbf{e}_{t_1}`
    and :math:`\textbf{e}_h + \textbf{e}_r \approx \textbf{e}_{t_2}` which results in
    :math:`\textbf{e}_{t_1} \approx \textbf{e}_{t_2}`.
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
        :param scoring_fct_norm: The :math:`l_p` norm applied in the interaction function. Is usually ``1`` or ``2.``.
        :param entity_initializer: Entity initializer function.
        :param entity_constrainer: Entity constrainer function.
        :param relation_initializer: Relation initializer function.
        :param relation_constrainer: Relation constrainer function. Defaults to none.
        :param kwargs:
            Remaining keyword arguments to forward to :meth:`pykeen.models.ERModel.__init__`
        :param regularizer:
            a regularizer, or a hint thereof. Used for both, entity and relation representations;
            directly use :class:`ERModel` if you need more flexibility
        :param regularizer_kwargs:
            keyword-based parameters for the regularizer

        .. seealso::

           - OpenKE `implementation of TransE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
        """
        super().__init__(
            interaction=TransEInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm),
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
