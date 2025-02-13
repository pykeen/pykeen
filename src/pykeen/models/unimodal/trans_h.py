"""An implementation of TransH."""

import itertools
from collections.abc import Mapping
from typing import Any, ClassVar

from class_resolver import HintOrType, OptionalKwargs
from torch.nn import functional, init

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import TransHInteraction
from ...regularizers import NormLimitRegularizer, OrthogonalityRegularizer, Regularizer
from ...typing import FloatTensor, Hint, Initializer

__all__ = [
    "TransH",
]


class TransH(ERModel[FloatTensor, tuple[FloatTensor, FloatTensor], FloatTensor]):
    r"""An implementation of TransH [wang2014]_.

    This model represents entities as $d$-dimensional vectors,
    and relations as pair of a normal vector and translation inside the hyperplane.
    They are stored in an :class:`~pykeen.nn.representation.Embedding`. The representations are then passed
    to the :class:`~pykeen.nn.modules.TransHInteraction` function to obtain scores.

    .. seealso::

       - OpenKE `implementation of TransH <https://github.com/thunlp/OpenKE/blob/master/models/TransH.py>`_
    ---
    citation:
        author: Wang
        year: 2014
        link: https://aaai.org/papers/8870-knowledge-graph-embedding-by-translating-on-hyperplanes/
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )
    #: The custom regularizer used by [wang2014]_ for TransH
    regularizer_default: ClassVar[type[Regularizer]] = NormLimitRegularizer
    #: The settings used by [wang2014]_ for TransH
    # The regularization in TransH enforces the defined soft constraints that should computed only for every batch.
    # Therefore, apply_only_once is always set to True.
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.05, apply_only_once=True, dim=-1, p=2, power_norm=True, max_norm=1.0
    )
    #: The custom regularizer used by [wang2014]_ for TransH
    relation_regularizer_default: ClassVar[type[Regularizer]] = OrthogonalityRegularizer
    #: The settings used by [wang2014]_ for TransH
    relation_regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.05, apply_only_once=True, epsilon=1e-5
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 2,
        power_norm: bool = False,
        entity_initializer: Hint[Initializer] = init.xavier_normal_,
        # note: this parameter is not named "entity_regularizer" for compatability with the
        #       regularizer-specific HPO code
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        relation_initializer: Hint[Initializer] = init.xavier_normal_,
        relation_regularizer: HintOrType[Regularizer] = None,
        relation_regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        r"""Initialize TransH.

        :param embedding_dim:
            the entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm:
            The norm used with :func:`torch.linalg.vector_norm`. Typically is 1 or 2.
        :param power_norm:
            Whether to use the p-th power of the $L_p$ norm. It has the advantage of being differentiable around 0,
            and numerically more stable.

        :param entity_initializer:
            The entity initializer function. Defaults to :func:`pykeen.nn.init.xavier_normal_`.
        :param regularizer:
            The entity regularizer. Defaults to :attr:`pykeen.models.TransH.regularizer_default`.
        :param regularizer_kwargs:
            Keyword-based parameters for the entity regularizer. If `entity_regularizer` is None,
            the default from :attr:`pykeen.models.TransH.regularizer_default_kwargs` will be used instead

        :param relation_initializer:
            The relation initializer function. Defaults to :func:`pykeen.nn.init.xavier_normal_`.
        :param relation_regularizer:
            The relation regularizer. Defaults to :attr:`pykeen.models.TransH.relation_regularizer_default`.
        :param relation_regularizer_kwargs:
            Keyword-based parameters for the relation regularizer. If `relation_regularizer` is None,
            the default from :attr:`pykeen.models.TransH.relation_regularizer_default_kwargs` will be used instead

        :param kwargs:
            Remaining keyword arguments are passed to :class:`~pykeen.models.ERModel`
        """
        super().__init__(
            interaction=TransHInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm, power_norm=power_norm),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=[
                # translation vector in hyperplane
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                ),
                # normal vector of hyperplane
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    # normalise the normal vectors to unit l2 length
                    constrainer=functional.normalize,
                ),
            ],
            **kwargs,
        )

        # As described in [wang2014], all entities and relations are used to compute the regularization term
        # which enforces the defined soft constraints.
        # thus, we need to use a weight regularizer instead of having an Embedding regularizer,
        # which only regularizes the weights used in a batch
        self.append_weight_regularizer(
            parameter=self.entity_representations[0].parameters(),
            regularizer=regularizer,
            regularizer_kwargs=regularizer_kwargs,
            # note: the following is already the default
            # default_regularizer=self.regularizer_default,
            # default_regularizer_kwargs=self.regularizer_default_kwargs,
        )
        self.append_weight_regularizer(
            parameter=itertools.chain.from_iterable(r.parameters() for r in self.relation_representations),
            regularizer=relation_regularizer,
            regularizer_kwargs=relation_regularizer_kwargs,
            default_regularizer=self.relation_regularizer_default,
            default_regularizer_kwargs=self.relation_regularizer_default_kwargs,
        )
