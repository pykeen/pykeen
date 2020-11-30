# -*- coding: utf-8 -*-

"""TransE."""

from typing import Optional

from torch.nn import functional

from ..base import ERModel
from ...losses import Loss
from ...nn import EmbeddingSpecification
from ...nn.init import xavier_uniform_
from ...nn.modules import TransEInteraction
from ...triples import TriplesFactory
from ...typing import DeviceHint
from ...utils import compose

__all__ = [
    'TransE',
]


class TransE(ERModel):
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
        scoring_fct_norm: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        r"""Initialize TransE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The :math:`l_p` norm applied in the interaction function. Is usually ``1`` or ``2.``.

        .. seealso::

           - OpenKE `implementation of TransE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
        """
        super().__init__(
            triples_factory=triples_factory,
            interaction=TransEInteraction(p=scoring_fct_norm, power_norm=False),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=xavier_uniform_,
                constrainer=functional.normalize,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=compose(
                    xavier_uniform_,
                    functional.normalize,
                ),
            ),
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )
