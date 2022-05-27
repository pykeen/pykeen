# -*- coding: utf-8 -*-

"""TransE."""

from typing import Any, ClassVar, Mapping

import torch.autograd
from torch import linalg
from torch.nn import functional

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.init import xavier_uniform_, xavier_uniform_norm_
from ...typing import Constrainer, Hint, Initializer

__all__ = [
    "TransE",
]


class TransE(EntityRelationEmbeddingModel):
    r"""An implementation of TransE [bordes2013]_.

    TransE models relations as a translation from head to tail entities in :math:`\textbf{e}`:

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
    ---
    citation:
        author: Bordes
        year: 2013
        link: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 1,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = None,
        **kwargs,
    ) -> None:
        r"""Initialize TransE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The :math:`l_p` norm applied in the interaction function. Is usually ``1`` or ``2.``.
        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`
        :param entity_constrainer: Entity constrainer function. Defaults to :func:`torch.nn.init.normalize`
        :param relation_initializer: Relation initializer function.
            Defaults to :func:`pykeen.nn.init.xavier_uniform_norm_`
        :param relation_constrainer: Relation constrainer function. Defaults to none.
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.EntityRelationEmbeddingModel`

        .. seealso::

           - OpenKE `implementation of TransE <https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py>`_
        """
        super().__init__(
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
            ),
            **kwargs,
        )
        self.scoring_fct_norm = scoring_fct_norm

    # docstr-coverage: inherited
    def score_hrt(self, hrt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0])
        r = self.relation_embeddings(indices=hrt_batch[:, 1])
        t = self.entity_embeddings(indices=hrt_batch[:, 2])

        # TODO: Use torch.cdist
        #  There were some performance/memory issues with cdist, cf.
        #  https://github.com/pytorch/pytorch/issues?q=cdist however, @mberr thinks
        #  they are mostly resolved by now. A Benefit would be that we can harness the
        #  future (performance) improvements made by the core torch developers. However,
        #  this will require some benchmarking.
        return -linalg.vector_norm(h + r - t, dim=-1, ord=self.scoring_fct_norm, keepdim=True)

    # docstr-coverage: inherited
    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0])
        r = self.relation_embeddings(indices=hr_batch[:, 1])
        t = self.entity_embeddings(indices=None)

        # TODO: Use torch.cdist (see note above in score_hrt())
        return -linalg.vector_norm(h[:, None, :] + r[:, None, :] - t[None, :, :], dim=-1, ord=self.scoring_fct_norm)

    # docstr-coverage: inherited
    def score_h(self, rt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=None)
        r = self.relation_embeddings(indices=rt_batch[:, 0])
        t = self.entity_embeddings(indices=rt_batch[:, 1])

        # TODO: Use torch.cdist (see note above in score_hrt())
        return -linalg.vector_norm(h[None, :, :] + (r[:, None, :] - t[:, None, :]), dim=-1, ord=self.scoring_fct_norm)
