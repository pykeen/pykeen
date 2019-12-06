# -*- coding: utf-8 -*-

"""Implementation of NTN."""

from typing import Optional

import torch
from torch import nn

from ..base import BaseModule
from ...losses import Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...utils import get_embedding_in_canonical_shape

__all__ = [
    'NTN',
]


class NTN(BaseModule):
    """An implementation of NTN from [socher2013]_.

    In NTN, a bilinear tensor layer relates the two entity vectors across multiple dimensions.

    Scoring function:
        u_R.T . f(h.T . W_R^[1:k] . t + V_r . [h; t] + b_R)

    where h.T . W_R^[1:k] . t denotes the bilinear tensor product.

    .. seealso::

       - Original Implementation (Matlab): `<https://github.com/khurram18/NeuralTensorNetworks>`_
       - TensorFlow: `<https://github.com/dddoss/tensorflow-socher-ntn>`_
       - Keras: `<https://github.com/dapurv5/keras-neural-tensor-layer (Keras)>`_
    """

    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=350, q=25),
        num_slices=dict(type=int, low=2, high=4),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 100,
        num_slices: int = 4,
        entity_embeddings: Optional[nn.Embedding] = None,
        w: Optional[nn.Embedding] = None,
        vh: Optional[nn.Embedding] = None,
        vt: Optional[nn.Embedding] = None,
        b: Optional[nn.Embedding] = None,
        u: Optional[nn.Embedding] = None,
        criterion: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        non_linearity: Optional[nn.Module] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model."""
        if non_linearity is None:
            non_linearity = nn.Tanh()

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            criterion=criterion,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        self.num_slices = num_slices

        self.w = w
        self.vh = vh
        self.vt = vt
        self.b = b
        self.u = u
        self.non_linearity = non_linearity

        # Finalize initialization
        self._init_weights_on_device()

    def init_empty_weights_(self):  # noqa: D102
        # Initialize entity embeddings
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        # bi-linear tensor layer
        # W_R: (d, d, k); store as (k, d, d)
        if self.w is None:
            init_w = torch.randn(self.num_relations, self.num_slices, self.embedding_dim, self.embedding_dim)
            self.w = nn.Parameter(init_w, requires_grad=True)
        # V_R: (k, 2d)
        if self.vh is None:
            init_vh = torch.randn(self.num_relations, self.num_slices, self.embedding_dim)
            self.vh = nn.Parameter(init_vh, requires_grad=True)
        if self.vt is None:
            init_vt = torch.randn(self.num_relations, self.num_slices, self.embedding_dim)
            self.vt = nn.Parameter(init_vt, requires_grad=True)
        # b_R: (k,)
        if self.b is None:
            init_b = torch.randn(self.num_relations, self.num_slices)
            self.b = nn.Parameter(init_b, requires_grad=True)
        # u_R: (k,)
        if self.u is None:
            init_u = torch.randn(self.num_relations, self.num_slices)
            self.u = nn.Parameter(init_u, requires_grad=True)

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.w = None
        self.vh = None
        self.b = None
        self.u = None
        return self

    def _score(
        self,
        h_ind: Optional[torch.LongTensor] = None,
        r_ind: Optional[torch.LongTensor] = None,
        t_ind: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute scores for NTN.

        :param h_ind: shape: (batch_size,)
        :param r_ind: shape: (batch_size,)
        :param t_ind: shape: (batch_size,)

        :return: shape: (batch_size, num_entities)
        """
        assert r_ind is not None

        #: shape: (batch_size, num_entities, d)
        h = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=h_ind)
        t = get_embedding_in_canonical_shape(embedding=self.entity_embeddings, ind=t_ind)

        #: Prepare h: (b, e, d) -> (b, e, 1, 1, d)
        h_for_w = h.unsqueeze(dim=-2).unsqueeze(dim=-2)

        #: Prepare t: (b, e, d) -> (b, e, 1, d, 1)
        t_for_w = t.unsqueeze(dim=-2).unsqueeze(dim=-1)

        #: Prepare w: (R, k, d, d) -> (b, k, d, d) -> (b, 1, k, d, d)
        w_r = self.w.index_select(dim=0, index=r_ind).unsqueeze(dim=1)

        # h.T @ W @ t, shape: (b, e, k, 1, 1)
        hwt = (h_for_w @ w_r @ t_for_w)

        #: reduce (b, e, k, 1, 1) -> (b, e, k)
        hwt = hwt.squeeze(dim=-1).squeeze(dim=-1)

        #: Prepare vh: (R, k, d) -> (b, k, d) -> (b, 1, k, d)
        vh_r = self.vh.index_select(dim=0, index=r_ind).unsqueeze(dim=1)

        #: Prepare h: (b, e, d) -> (b, e, d, 1)
        h_for_v = h.unsqueeze(dim=-1)

        # V_h @ h, shape: (b, e, k, 1)
        vhh = vh_r @ h_for_v

        #: reduce (b, e, k, 1) -> (b, e, k)
        vhh = vhh.squeeze(dim=-1)

        #: Prepare vt: (R, k, d) -> (b, k, d) -> (b, 1, k, d)
        vt_r = self.vt.index_select(dim=0, index=r_ind).unsqueeze(dim=1)

        #: Prepare t: (b, e, d) -> (b, e, d, 1)
        t_for_v = t.unsqueeze(dim=-1)

        # V_t @ t, shape: (b, e, k, 1)
        vtt = vt_r @ t_for_v

        #: reduce (b, e, k, 1) -> (b, e, k)
        vtt = vtt.squeeze(dim=-1)

        #: Prepare b: (R, k) -> (b, k) -> (b, 1, k)
        b = self.b.index_select(dim=0, index=r_ind).unsqueeze(dim=1)

        # a = f(h.T @ W @ t + Vh @ h + Vt @ t + b), shape: (b, e, k)
        pre_act = hwt + vhh + vtt + b
        act = self.non_linearity(pre_act)

        # prepare u: (R, k) -> (b, k) -> (b, 1, k, 1)
        u = self.u.index_select(dim=0, index=r_ind).unsqueeze(dim=1).unsqueeze(dim=-1)

        # prepare act: (b, e, k) -> (b, e, 1, k)
        act = act.unsqueeze(dim=-2)

        # compute score, shape: (b, e, 1, 1)
        score = act @ u

        # reduce
        score = score.squeeze(dim=-1).squeeze(dim=-1)

        return score

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hrt_batch[:, 0], r_ind=hrt_batch[:, 1], t_ind=hrt_batch[:, 2])

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_ind=hr_batch[:, 0], r_ind=hr_batch[:, 1])

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(r_ind=rt_batch[:, 0], t_ind=rt_batch[:, 1])
