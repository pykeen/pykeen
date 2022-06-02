# -*- coding: utf-8 -*-

"""Implementation of the HolE model."""

from typing import Any, ClassVar, Mapping, Optional

from class_resolver import Hint, OptionalKwargs

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import HolEInteraction
from ...nn.init import xavier_uniform_
from ...typing import Constrainer, Initializer
from ...utils import clamp_norm

__all__ = [
    "HolE",
]


class HolE(ERModel):
    r"""An implementation of HolE [nickel2016]_.

    Holographic embeddings (HolE) make use of the circular correlation operator to compute interactions between
    latent features of entities and relations:

    .. math::

        f(h,r,t) = \sigma(\textbf{r}^{T}(\textbf{h} \star \textbf{t}))

    where the circular correlation $\star: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^d$ is defined as:

    .. math::

        [\textbf{a} \star \textbf{b}]_i = \sum_{k=0}^{d-1} \textbf{a}_{k} * \textbf{b}_{(i+k)\ mod \ d}

    By using the correlation operator each component $[\textbf{h} \star \textbf{t}]_i$ represents a sum over a
    fixed partition over pairwise interactions. This enables the model to put semantic similar interactions into the
    same partition and share weights through $\textbf{r}$. Similarly irrelevant interactions of features could also
    be placed into the same partition which could be assigned a small weight in $\textbf{r}$.

    .. seealso::

       - `author's implementation of HolE <https://github.com/mnick/holographic-embeddings>`_
       - `scikit-kge implementation of HolE <https://github.com/mnick/scikit-kge>`_
       - OpenKE `implementation of HolE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
    ---
    citation:
        author: Nickel
        year: 2016
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828
        github: mnick/holographic-embeddings
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
