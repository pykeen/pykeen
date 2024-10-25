"""Implementation of the HolE model."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional

from class_resolver import Hint, OptionalKwargs

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import HolEInteraction
from ...nn.init import xavier_uniform_
from ...typing import Constrainer, FloatTensor, Initializer
from ...utils import clamp_norm

__all__ = [
    "HolE",
]


class HolE(ERModel[FloatTensor, FloatTensor, FloatTensor]):
    r"""An implementation of HolE from [nickel2016]_.

    This model represents both entities and relations as $d$-dimensional vectors stored in an
    :class:`~pykeen.nn.representation.Embedding` matrix.
    The representations are then passed to the :class:`~pykeen.nn.modules.HolEInteraction` function to obtain
    scores.

    .. note ::
        The original paper describes modelling the probability as $\sigma(f(h, r, t))$, however, since the margin
        ranking loss is used for all experiments, the implementation of the score function does not include the
        $\sigma$.

    .. seealso::

       - `author's implementation of HolE <https://github.com/mnick/holographic-embeddings>`_
       - `scikit-kge implementation of HolE <https://github.com/mnick/scikit-kge>`_
       - OpenKE `implementation of HolE <https://github.com/thunlp/OpenKE/blob/64c4ec8157abfc39112772e9825a349091da45f1/openke/module/model/HolE.py>`_
    ---
    citation:
        author: Nickel
        year: 2016
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828
        github: mnick/holographic-embeddings
        arxiv: 1510.04935
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    #: The default settings for the entity constrainer
    entity_constrainer_default_kwargs = dict(maxnorm=1.0, p=2, dim=-1)

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        # Initialisation, cf. https://github.com/mnick/scikit-kge/blob/master/skge/param.py#L18-L27
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = clamp_norm,  # type: ignore
        entity_constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_representation_kwargs: OptionalKwargs = None,
        relation_initializer: Hint[Constrainer] = xavier_uniform_,
        relation_representation_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            the embedding dimension (for entities and relations)

        :param entity_initializer:
            the initializer for entity representations
        :param entity_constrainer:
            the constrainer for entity representations
        :param entity_constrainer_kwargs:
            keyword-based parameters passed to the constrainer. If None, use :attr:`entity_constrainer_default_kwargs`
        :param entity_representation_kwargs:
            additional keyword-based parameters passed to the entity representation

        :param relation_initializer:
            the initializer for relation representations
        :param relation_representation_kwargs:
            additional keyword-based parameters passed to the entity representation

        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        super().__init__(
            interaction=HolEInteraction,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                constrainer_kwargs=entity_constrainer_kwargs or self.entity_constrainer_default_kwargs,
                **(entity_representation_kwargs or {}),
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                **(relation_representation_kwargs or {}),
            ),
            **kwargs,
        )
