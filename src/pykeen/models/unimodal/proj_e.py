"""Implementation of ProjE."""

from collections.abc import Mapping
from typing import Any, ClassVar

from class_resolver import HintOrType, OptionalKwargs, ResolverKey, update_docstring_with_resolver_keys
from torch import nn

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEWithLogitsLoss, Loss
from ...nn.init import xavier_uniform_
from ...nn.modules import ProjEInteraction
from ...typing import Hint, Initializer

__all__ = [
    "ProjE",
]


class ProjE(ERModel):
    r"""An implementation of ProjE from [shi2017]_.

    ProjE represents entities and relations using a $d$-dimensional embedding vector stored in an
    :class:`~pykeen.nn.representation.Embedding`. On top of these representations, this model uses the
    :class:`~pykeen.nn.modules.ProjEInteraction` to calculate scores.

    .. seealso::

       - Official Implementation: https://github.com/nddsg/ProjE
    ---
    citation:
        author: Shi
        year: 2017
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14279
        github: nddsg/ProjE
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[type[Loss]] = BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(reduction="mean")

    @update_docstring_with_resolver_keys(
        ResolverKey(name="inner_non_linearity", resolver="class_resolver.contrib.torch.activation_resolver")
    )
    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        inner_non_linearity: HintOrType[nn.Module] = None,
        inner_non_linearity_kwargs: OptionalKwargs = None,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = xavier_uniform_,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            the embedding dimension
        :param inner_non_linearity:
            the inner non-linearity, of a hint thereof. cf. :class:`pykeen.nn.modules.ProjEInteraction`
        :param inner_non_linearity_kwargs:
            additional keyword-based parameters used to instantiate the non-linearity.
        :param entity_initializer:
            the entity representation initializer, defaults to :func:`~pykeen.nn.init.xavier_uniform_`.
        :param relation_initializer:
            the relation representation initializer, defaults to :func:`~pykeen.nn.init.xavier_uniform_`.
        :param kwargs:
            additional keyword-based parameters passed to :class:`~pykeen.models.ERModel`
        """
        super().__init__(
            interaction=ProjEInteraction,
            interaction_kwargs=dict(
                embedding_dim=embedding_dim,
                inner_activation=inner_non_linearity,
                inner_activation_kwargs=inner_non_linearity_kwargs,
            ),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
        )
