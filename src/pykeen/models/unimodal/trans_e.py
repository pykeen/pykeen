# -*- coding: utf-8 -*-

"""TransE."""

from typing import Optional, Tuple

import torch.autograd
from torch.nn import functional

from ..base import EntityRelationEmbeddingModel
from ...losses import Loss
from ...nn.init import xavier_uniform_
from ...nn.modules import TranslationalInteractionFunction
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import compose, get_hr_indices, get_hrt_indices, get_ht_indices, get_rt_indices

__all__ = [
    'TransE',
]


class TransE(EntityRelationEmbeddingModel):
    r"""TransE models relations as a translation from head to tail entities in :math:`\textbf{e}` [bordes2013]_.

    .. math::

        \textbf{e}_h + \textbf{e}_r \approx \textbf{e}_t

    This equation is rearranged and the :math:`l_p` norm is applied to create the TransE interaction function.

    .. math::

        f(h, r, t) = - \|\textbf{e}_h + \textbf{e}_r - \textbf{e}_t\|_{p}

    While this formulation is computationally efficient, it inherently cannot model one-to-many, many-to-one, and
    many-to-many relationships. For triples :math:`(h,r,t_1), (h,r,t_2) \in \mathcal{K}` where :math:`t_1 \neq t_2`,
    the model adapts the embeddings in order to ensure :math:`\textbf{e}_h + \textbf{e}_r \approx \textbf{e}_{t_1}`
    and :math:`\textbf{e}_h + \textbf{e}_r \approx \textbf{e}_{t_2}` which results in
    :math:`\textbf{e}_{t_1} \approx \textbf{e}_{t_2}`.
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        automatic_memory_optimization: Optional[bool] = None,
        scoring_fct_norm: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        r"""Initialize TransE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The :math:`l_p` norm applied in the interaction function. Is usually ``1`` or ``2.``.

        .. seealso::

           - OpenKE `implementation of TransE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            entity_initializer=xavier_uniform_,
            relation_initializer=compose(
                xavier_uniform_,
                functional.normalize,
            ),
            entity_constrainer=functional.normalize,
        )
        self.interaction_function = TranslationalInteractionFunction(p=scoring_fct_norm)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h_indices, r_indices, t_indices = get_hrt_indices(hrt_batch)
        h, r, t = self._get_hrt(h_indices=h_indices, r_indices=r_indices, t_indices=t_indices)
        return self.interaction_function(h=h, r=r, t=t).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h_indices, r_indices, t_indices = get_hr_indices(hr_batch)
        h, r, t = self._get_hrt(h_indices=h_indices, r_indices=r_indices, t_indices=t_indices)
        return self.interaction_function(h=h, r=r, t=t).view(hr_batch.shape[0], self.num_entities)

    def score_r(self, ht_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h_indices, r_indices, t_indices = get_ht_indices(ht_batch)
        h, r, t = self._get_hrt(h_indices=h_indices, r_indices=r_indices, t_indices=t_indices)
        return self.interaction_function(h=h, r=r, t=t)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h_indices, r_indices, t_indices = get_rt_indices(rt_batch)
        h, r, t = self._get_hrt(h_indices=h_indices, r_indices=r_indices, t_indices=t_indices)
        return self.interaction_function(h=h, r=r, t=t).view(rt_batch.shape[0], self.num_entities)

    def _get_hrt(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        h = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        r = self.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        t = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)
        return h, r, t
