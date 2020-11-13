# -*- coding: utf-8 -*-

"""An implementation of TransH."""

from typing import Optional

import torch
from torch.nn import functional

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...nn import Embedding, functional as pkf
from ...nn.emb import EmbeddingSpecification
from ...regularizers import Regularizer, TransHRegularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'TransH',
]


class TransH(EntityRelationEmbeddingModel):
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
        embedding_dim=dict(type=int, low=50, high=300, q=50),
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
        automatic_memory_optimization: Optional[bool] = None,
        scoring_fct_norm: int = 2,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize TransH.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The :math:`l_p` norm applied in the interaction function. Is usually ``1`` or ``2.``.
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )
        self.scoring_fct_norm = scoring_fct_norm

        # embeddings
        self.normal_vector_embeddings = Embedding.from_specification(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=embedding_dim,
            specification=EmbeddingSpecification(
                # Normalise the normal vectors by their l2 norms
                constrainer=functional.normalize,
            ),
        )

    def post_parameter_update(self) -> None:  # noqa: D102
        super().post_parameter_update()
        self.normal_vector_embeddings.post_parameter_update()

    def regularize_if_necessary(self) -> None:
        """Update the regularizer's term given some tensors, if regularization is requested."""
        # As described in [wang2014], all entities and relations are used to compute the regularization term
        # which enforces the defined soft constraints.
        super().regularize_if_necessary(
            self.entity_embeddings(indices=None),
            self.normal_vector_embeddings(indices=None),  # FIXME
            self.relation_embeddings(indices=None),
        )

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        d_r = self.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        w_r = self.normal_vector_embeddings.get_in_canonical_shape(indices=r_indices)
        t = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)
        self.regularize_if_necessary()
        return pkf.transh_interaction(h, w_r, d_r, t, p=self.scoring_fct_norm)
