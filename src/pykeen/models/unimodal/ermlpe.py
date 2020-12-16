# -*- coding: utf-8 -*-

"""An implementation of the extension to ERMLP."""

from typing import Any, ClassVar, Mapping, Optional, Type

from ..base import ERModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn import EmbeddingSpecification
from ...nn.modules import ERMLPEInteraction
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'ERMLPE',
]


class ERMLPE(ERModel):
    r"""An extension of ERMLP proposed by [sharifzadeh2019]_.

    This model uses a neural network-based approach similar to ERMLP and with slight modifications.
    In ERMLP, the model is:

    .. math::

        f(h, r, t) = \textbf{w}^{T} g(\textbf{W} [\textbf{h}; \textbf{r}; \textbf{t}])

    whereas in ERMPLE the model is:

    .. math::

        f(h, r, t) = \textbf{t}^{T} f(\textbf{W} (g(\textbf{W} [\textbf{h}; \textbf{r}]))

    including dropouts and batch-norms between each two hidden layers.
    ConvE can be seen as a special case of ERMLPE that contains the unnecessary inductive bias of convolutional
    filters. The aim of this model is to show that lifting this bias from ConvE (which simply leaves us with a
    modified ERMLP model), not only reduces the number of parameters but also improves performance.

    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        hidden_dim=dict(type=int, low=5, high=9, scale='power_two'),
        input_dropout=DEFAULT_DROPOUT_HPO_RANGE,
        hidden_dropout=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        triples_factory: TriplesFactory,
        hidden_dim: int = 300,
        input_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        embedding_dim: int = 200,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            interaction=ERMLPEInteraction(
                hidden_dim=hidden_dim,
                input_dropout=input_dropout,
                hidden_dropout=hidden_dropout,
                embedding_dim=embedding_dim,
            ),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
            ),
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
