# -*- coding: utf-8 -*-

"""Implementation of ProjE."""

from typing import Any, ClassVar, Mapping, Optional, Type

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

    ProjE is a neural network-based approach with a *combination* and a *projection* layer. The interaction model
    first combines $h$ and $r$ by following combination operator:

    .. math::

        \textbf{h} \otimes \textbf{r} = \textbf{D}_e \textbf{h} + \textbf{D}_r \textbf{r} + \textbf{b}_c

    where $\textbf{D}_e, \textbf{D}_r \in \mathbb{R}^{k \times k}$ are diagonal matrices which are used as shared
    parameters among all entities and relations, and $\textbf{b}_c \in \mathbb{R}^{k}$ represents the candidate bias
    vector shared across all entities. Next, the score for the triple $(h,r,t) \in \mathbb{K}$ is computed:

    .. math::

        f(h, r, t) = g(\textbf{t} \ z(\textbf{h} \otimes \textbf{r}) + \textbf{b}_p)

    where $g$ and $z$ are activation functions, and $\textbf{b}_p$ represents the shared projection bias vector.

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
    loss_default: ClassVar[Type[Loss]] = BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(reduction="mean")

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        inner_non_linearity: Optional[nn.Module] = None,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = xavier_uniform_,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            the embedding dimension
        :param inner_non_linearity:
            the inner non-linearity, of a hint thereof. cf. :meth:`ProjEInteraction.__init__`
        :param entity_initializer:
            the entity representation initializer
        :param relation_initializer:
            the relation representation initializer
        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        super().__init__(
            interaction=ProjEInteraction,
            interaction_kwargs=dict(
                embedding_dim=embedding_dim,
                inner_non_linearity=inner_non_linearity,
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
