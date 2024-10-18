"""Implementation of TransR."""

from collections.abc import Mapping
from typing import Any, ClassVar

import torch
import torch.autograd
import torch.nn.init
from class_resolver import OptionalKwargs

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import TransRInteraction
from ...nn.init import xavier_uniform_, xavier_uniform_norm_
from ...typing import Constrainer, FloatTensor, Hint, Initializer
from ...utils import clamp_norm

__all__ = [
    "TransR",
]


class TransR(ERModel[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    r"""An implementation of TransR from [lin2015]_.

    This model represents entities as $d$-dimensional vectors, and relations as $k$-dimensional vectors.
    To bring them into the same vector space, a relation-specific projection is learned, too.
    All representations are stored in :class:`~pykeen.nn.representation.Embedding` matrices.

    The representations are then passed to the :class:`~pykeen.nn.modules.TransRInteraction` function to obtain scores.

    The following constraints are applied:

        - $\|\textbf{e}_h\|_2 \leq 1$
        - $\|\textbf{r}_r\|_2 \leq 1$
        - $\|\textbf{e}_t\|_2 \leq 1$

    as well as inside the :class:`~pykeen.nn.modules.TransRInteraction`

        - $\|\textbf{M}_{r}\textbf{e}_h\|_2 \leq 1$
        - $\|\textbf{M}_{r}\textbf{e}_t\|_2 \leq 1$

    .. seealso::

       - OpenKE `TensorFlow implementation of TransR
         <https://github.com/thunlp/OpenKE/blob/master/models/TransR.py>`_
       - OpenKE `PyTorch implementation of TransR
         <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransR.py>`_
    ---
    citation:
        author: Lin
        year: 2015
        link: http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        relation_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        relation_dim: int = 30,
        max_projection_norm: float = 1.0,
        # interaction function kwargs
        scoring_fct_norm: int = 1,
        power_norm: bool = False,
        # entity embedding
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_initializer_kwargs: OptionalKwargs = None,
        entity_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        # relation embedding
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_initializer_kwargs: OptionalKwargs = None,
        relation_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        # relation projection
        relation_projection_initializer: Hint[Initializer] = torch.nn.init.xavier_uniform_,
        relation_projection_initializer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """Initialize the model.

        :param embedding_dim: The entity embedding dimension $d$.
        :param relation_dim: The relation embedding dimension $k$.
        :param max_projection_norm:
            The maximum norm to be clamped after projection.

        :param scoring_fct_norm:
            The norm used with :func:`torch.linalg.vector_norm`. Typically is 1 or 2.
        :param power_norm:
            Whether to use the p-th power of the $L_p$ norm. It has the advantage of being differentiable around 0,
            and numerically more stable.

        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`.
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer.
        :param entity_constrainer: The entity constrainer. Defaults to :func:`pykeen.utils.clamp_norm`.

        :param relation_initializer:
            Relation initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_norm_`.
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer.
        :param relation_constrainer: The relation constrainer. Defaults to :func:`pykeen.utils.clamp_norm`.

        :param relation_projection_initializer:
            Relation projection initializer function. Defaults to :func:`torch.nn.init.xavier_uniform_`.
        :param relation_projection_initializer_kwargs:
            Keyword arguments to be used when calling the relation projection initializer.

        :param kwargs: Remaining keyword arguments passed through to :class:`~pykeen.models.ERModel`.
        """
        # TODO: Initialize from TransE
        super().__init__(
            interaction=TransRInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm, power_norm=power_norm),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                initializer_kwargs=entity_initializer_kwargs,
                constrainer=entity_constrainer,
                constrainer_kwargs=dict(maxnorm=max_projection_norm, p=scoring_fct_norm, dim=-1),
            ),
            relation_representations_kwargs=[
                # relation embedding
                dict(
                    shape=(relation_dim,),
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs,
                    constrainer=relation_constrainer,
                    constrainer_kwargs=dict(maxnorm=max_projection_norm, p=scoring_fct_norm, dim=-1),
                ),
                # relation projection
                dict(
                    shape=(embedding_dim, relation_dim),
                    initializer=relation_projection_initializer,
                    initializer_kwargs=relation_projection_initializer_kwargs,
                ),
            ],
            **kwargs,
        )
