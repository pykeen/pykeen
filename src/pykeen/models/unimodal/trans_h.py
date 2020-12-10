# -*- coding: utf-8 -*-

"""An implementation of TransH."""

from typing import Optional

from torch.nn import functional

from ..base import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss
from ...nn import EmbeddingSpecification
from ...nn.modules import TransHInteraction
from ...regularizers import TransHRegularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import pop_only

__all__ = [
    'TransH',
]


class TransH(ERModel):
    r"""An implementation of TransH [wang2014]_.

    This model extends :class:`pykeen.models.TransE` by applying the translation from head to tail entity in a
    relational-specific hyperplane in order to address its inability to model one-to-many, many-to-one, and
    many-to-many relations.

    In TransH, each relation is represented by a hyperplane, or more specifically a normal vector of this hyperplane
    $\textbf{w}_{r} \in \mathbb{R}^d$ and a vector $\textbf{d}_{r} \in \mathbb{R}^d$ that lies in the hyperplane.
    To compute the plausibility of a triple $(h,r,t)\in \mathbb{K}$, the head embedding $\textbf{e}_h \in \mathbb{R}^d$
    and the tail embedding $\textbf{e}_t \in \mathbb{R}^d$ are first projected onto the relation-specific hyperplane:

    .. math::

        \textbf{e'}_{h,r} = \textbf{e}_h - \textbf{w}_{r}^\top \textbf{e}_h \textbf{w}_r

        \textbf{e'}_{t,r} = \textbf{e}_t - \textbf{w}_{r}^\top \textbf{e}_t \textbf{w}_r

    where $\textbf{h}, \textbf{t} \in \mathbb{R}^d$. Then, the projected embeddings are used to compute the score
    for the triple $(h,r,t)$:

    .. math::

        f(h, r, t) = -\|\textbf{e'}_{h,r} + \textbf{d}_r - \textbf{e'}_{t,r}\|_{p}^2

    .. seealso::

       - OpenKE `implementation of TransH <https://github.com/thunlp/OpenKE/blob/master/models/TransH.py>`_
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )
    #: The custom regularizer used by [wang2014]_ for TransH
    regularizer_default = TransHRegularizer
    #: The settings used by [wang2014]_ for TransH
    regularizer_default_kwargs = dict(
        weight=0.05,
        epsilon=1e-5,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 2,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        r"""Initialize TransH.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The :math:`l_p` norm applied in the interaction function. Is usually ``1`` or ``2.``.
        """
        super().__init__(
            triples_factory=triples_factory,
            interaction=TransHInteraction(
                p=scoring_fct_norm,
                power_norm=False,
            ),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
            ),
            relation_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                ),
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    # Normalise the normal vectors by their l2 norms
                    constrainer=functional.normalize,
                ),
            ],
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            predict_with_sigmoid=predict_with_sigmoid,
        )
        # Note that the TransH regularizer has a different interface
        self.regularizer = self._instantiate_default_regularizer(
            entity_embeddings=pop_only(self.entity_representations[0].parameters()),
            relation_embeddings=pop_only(self.relation_representations[0].parameters()),
            normal_vector_embeddings=pop_only(self.relation_representations[1].parameters()),
        )
