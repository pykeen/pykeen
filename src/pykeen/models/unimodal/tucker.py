"""Implementation of TuckEr."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

from class_resolver import OptionalKwargs

from ..nbase import ERModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn import TuckERInteraction
from ...nn.init import xavier_normal_
from ...typing import FloatTensor, Hint, Initializer

__all__ = [
    "TuckER",
]


class TuckER(ERModel[FloatTensor, FloatTensor, FloatTensor]):
    r"""An implementation of TuckEr from [balazevic2019]_.

    It represents entities by $d_e$-dimensional vectors and relations by $d_r$-dimensional vectors, stored in
    :class:`~pykeen.nn.representation.Embedding`. The state-ful :class:`~pykeen.nn.modules.TuckERInteraction` is then
    used to score triples.

    For $E$ entities and $R$ relations, the model has $Ed_e + Rd_r + d_e^2d_r$ effective parameters (ignoring additional
    parameters from the :class:`torch.nn.BatchNorm1d` layers in :class:`~pykeen.nn.modules.TuckERInteraction`).

    .. seealso::

       - Official implementation: https://github.com/ibalazevic/TuckER
       - pykg2vec implementation of TuckEr https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TuckER.py
    ---
    citation:
        author: Balažević
        year: 2019
        link: https://arxiv.org/abs/1901.09590
        github: ibalazevic/TuckER
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        relation_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        dropout_0=DEFAULT_DROPOUT_HPO_RANGE,
        dropout_1=DEFAULT_DROPOUT_HPO_RANGE,
        dropout_2=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[type[Loss]] = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        relation_dim: Optional[int] = None,
        dropout_0: float = 0.3,
        dropout_1: float = 0.4,
        dropout_2: float = 0.5,
        apply_batch_normalization: bool = True,
        entity_initializer: Hint[Initializer] = xavier_normal_,
        relation_initializer: Hint[Initializer] = xavier_normal_,
        core_tensor_initializer: Hint[Initializer] = None,
        core_tensor_initializer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            the (entity) embedding dimension
        :param relation_dim:
            the relation embedding dimension. Defaults to `embedding_dim`.
        :param dropout_0:
            the first dropout, cf. formula
        :param dropout_1:
            the second dropout, cf. formula
        :param dropout_2:
            the third dropout, cf. formula
        :param apply_batch_normalization:
            whether to apply batch normalization
        :param entity_initializer:
            the entity representation initializer
        :param relation_initializer:
            the relation representation initializer
        :param core_tensor_initializer:
            the core tensor initializer
        :param core_tensor_initializer_kwargs:
            keyword-based parameters passed to the core tensor initializer
        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        relation_dim = relation_dim or embedding_dim
        super().__init__(
            interaction=TuckERInteraction,
            interaction_kwargs=dict(
                embedding_dim=embedding_dim,
                relation_dim=relation_dim,
                head_dropout=dropout_0,  # TODO: rename
                relation_dropout=dropout_1,
                head_relation_dropout=dropout_2,
                apply_batch_normalization=apply_batch_normalization,
                core_initializer=core_tensor_initializer,
                core_initializer_kwargs=core_tensor_initializer_kwargs,
            ),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations_kwargs=dict(
                shape=relation_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
        )
