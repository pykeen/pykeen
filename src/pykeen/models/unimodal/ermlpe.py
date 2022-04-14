# -*- coding: utf-8 -*-

"""An implementation of the extension to ERMLP."""

from typing import Any, ClassVar, Mapping, Optional, Type

from torch.nn.init import uniform_

from ..nbase import ERModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn.modules import ERMLPEInteraction
from ...typing import Hint, Initializer

__all__ = [
    "ERMLPE",
]


class ERMLPE(ERModel):
    r"""An extension of :class:`pykeen.models.ERMLP` proposed by [sharifzadeh2019]_.

    This model uses a neural network-based approach similar to ER-MLP and with slight modifications.
    In ER-MLP, the model is:

    .. math::

        f(h, r, t) = \textbf{w}^{T} g(\textbf{W} [\textbf{h}; \textbf{r}; \textbf{t}])

    whereas in ER-MLP (E) the model is:

    .. math::

        f(h, r, t) = \textbf{t}^{T} f(\textbf{W} (g(\textbf{W} [\textbf{h}; \textbf{r}]))

    including dropouts and batch-norms between each two hidden layers.
    ConvE can be seen as a special case of ER-MLP (E) that contains the unnecessary inductive bias of convolutional
    filters. The aim of this model is to show that lifting this bias from :class:`pykeen.models.ConvE` (which simply
    leaves us with a modified ER-MLP model), not only reduces the number of parameters but also improves performance.
    ---
    name: ER-MLP (E)
    citation:
        author: Sharifzadeh
        year: 2019
        link: https://github.com/pykeen/pykeen
        github: pykeen/pykeen
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        hidden_dim=dict(type=int, low=5, high=9, scale="power_two"),
        input_dropout=DEFAULT_DROPOUT_HPO_RANGE,
        hidden_dropout=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        hidden_dim: Optional[int] = None,
        input_dropout: float = 0.2,
        hidden_dropout: Optional[float] = None,
        entity_initializer: Hint[Initializer] = uniform_,
        relation_initializer: Hint[Initializer] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            the embedding dimension (for both, entities and relations)
        :param hidden_dim:
            the hidden dimension of the MLP; defaults to ``embedding_dim``.
        :param input_dropout:
            the input dropout of the MLP
        :param hidden_dropout:
            the hidden dropout of the MLP; defaults to ``input_dropout``.
        :param entity_initializer:
            the entity embedding initializer
        :param relation_initializer:
            the relation embedding initializer; defaults to ``entity_initializer``.
        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        super().__init__(
            interaction=ERMLPEInteraction,
            interaction_kwargs=dict(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                input_dropout=input_dropout,
                hidden_dropout=hidden_dropout,
            ),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer or entity_initializer,
            ),
            **kwargs,
        )
