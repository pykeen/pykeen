"""Implementation of DistMult."""

from collections.abc import Mapping
from typing import Any, ClassVar

from class_resolver import HintOrType, OptionalKwargs, ResolverKey, update_docstring_with_resolver_keys
from torch.nn import functional

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.init import xavier_normal_norm_, xavier_uniform_
from ...nn.modules import DistMultInteraction
from ...regularizers import LpRegularizer, Regularizer
from ...typing import Constrainer, FloatTensor, Hint, Initializer

__all__ = [
    "DistMult",
]


class DistMult(ERModel[FloatTensor, FloatTensor, FloatTensor]):
    r"""An implementation of DistMult from [yang2014]_.

    In this work, both entities and relations are represented by $d$-dimensional vectors stored in an
    :class:`~pykeen.nn.representation.Embedding` matrix.
    The entity representation vectors are further constrained to have unit $L_2$ norm.
    For the relation representations, a (soft) regularization term on the vector $L_2$ norm is used instead.

    The representations are then passed to the :class:`~pykeen.nn.modules.DistMultInteraction` function to obtain
    scores.

    This DistMult model can be seen as a simplification of the :class:`~pykeen.models.RESCAL` model,
    where the relation matrices are restricted to diagonal matrices:
    Because of its restriction to diagonal matrices, :class:`~pykeen.models.DistMult` is computationally cheaper than
    :class:`~pykeen.models.RESCAL`, but at the same time it is less expressive. For example, it is not able to
    model anti-symmetric relations.

    .. seealso::

       - OpenKE `implementation of DistMult <https://github.com/thunlp/OpenKE/blob/master/models/DistMult.py>`_
    ---
    citation:
        author: Yang
        year: 2014
        link: https://arxiv.org/abs/1412.6575
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The regularizer used by [yang2014]_ for DistMult
    #: In the paper, they use weight of 0.0001, mini-batch-size of 10, and dimensionality of vector 100
    #: Thus, when we use normalized regularization weight, the normalization factor is 10*sqrt(100) = 100, which is
    #: why the weight has to be increased by a factor of 100 to have the same configuration as in the paper.
    regularizer_default: ClassVar[type[Regularizer]] = LpRegularizer
    #: The LP settings used by [yang2014]_ for DistMult
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.1,
        p=2.0,
        normalize=True,
    )

    @update_docstring_with_resolver_keys(
        ResolverKey(name="regularizer", resolver="pykeen.regularizers.regularizer_resolver")
    )
    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        relation_initializer: Hint[Initializer] = xavier_normal_norm_,
        regularizer: HintOrType[Regularizer] = LpRegularizer,
        regularizer_kwargs: OptionalKwargs = None,
        entity_representations_kwargs: OptionalKwargs = None,
        relation_representations_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        r"""Initialize DistMult.

        :param embedding_dim:
            The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param entity_initializer:
            The method used to initialize the entity embedding. Defaults to Xavier/Glorot uniform, c.f.
            `OpenKE <https://github.com/thunlp/OpenKE/blob/adeed2c0d2bef939807ed4f69c1ea4db35fd149b/models/DistMult.py#L16-L17>`_
        :param entity_constrainer:
            The constrainer for entity embeddings. Defaults to unit L2 norm.

        :param relation_initializer:
            The method used to initialize the relation embedding. Defaults to using Xavier/Glorot uniform first and
            then normalizing te unit L2 length.
        :param regularizer:
            The *relation* representation regularizer.
        :param regularizer_kwargs:
            Additional keyword-based parameters. Defaults to :attr:`DistMult.regularizer_default_kwargs` for the
            default regularizer.

        :param entity_representations_kwargs:
            Additional parameters to ``entity_representations_kwargs`` passed to :class:`pykeen.models.ERModel`.
            Note that those take precedence of those which are filled in by this class.
        :param relation_representations_kwargs:
            Additional parameters to ``relation_representations_kwargs`` passed to :class:`pykeen.models.ERModel`.
            Note that those take precedence of those which are filled in by this class.
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.ERModel`
        """
        if regularizer is LpRegularizer and regularizer_kwargs is None:
            regularizer_kwargs = DistMult.regularizer_default_kwargs
        resolved_entity_representations_kwargs = dict(
            shape=embedding_dim,
            initializer=entity_initializer,
            constrainer=entity_constrainer,
            # note: DistMult only regularizes the relation embeddings;
            #       entity embeddings are hard constrained instead
        )
        resolved_entity_representations_kwargs.update(entity_representations_kwargs or {})
        resolved_relation_representations_kwargs = dict(
            shape=embedding_dim,
            initializer=relation_initializer,
            regularizer=regularizer,
            regularizer_kwargs=regularizer_kwargs,
        )
        resolved_relation_representations_kwargs.update(relation_representations_kwargs or {})
        super().__init__(
            interaction=DistMultInteraction,
            entity_representations_kwargs=resolved_entity_representations_kwargs,
            relation_representations_kwargs=resolved_relation_representations_kwargs,
            **kwargs,
        )
