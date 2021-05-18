# -*- coding: utf-8 -*-

"""Implementation of NTN."""

from typing import Any, ClassVar, Mapping, Optional

import torch
from torch import nn

from ..base import EntityEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn.emb import EmbeddingSpecification
from ...typing import Hint, Initializer

__all__ = [
    'NTN',
]


class NTN(EntityEmbeddingModel):
    r"""An implementation of NTN from [socher2013]_.

    NTN uses a bilinear tensor layer instead of a standard linear neural network layer:

    .. math::

        f(h,r,t) = \textbf{u}_{r}^{T} \cdot \tanh(\textbf{h} \mathfrak{W}_{r} \textbf{t}
        + \textbf{V}_r [\textbf{h};\textbf{t}] + \textbf{b}_r)

    where $\mathfrak{W}_r \in \mathbb{R}^{d \times d \times k}$ is the relation specific tensor, and the weight
    matrix $\textbf{V}_r \in \mathbb{R}^{k \times 2d}$, and the bias vector $\textbf{b}_r$ and
    the weight vector $\textbf{u}_r \in \mathbb{R}^k$ are the standard
    parameters of a neural network, which are also relation specific. The result of the tensor product
    $\textbf{h} \mathfrak{W}_{r} \textbf{t}$ is a vector $\textbf{x} \in \mathbb{R}^k$ where each entry $x_i$ is
    computed based on the slice $i$ of the tensor $\mathfrak{W}_{r}$:
    $\textbf{x}_i = \textbf{h}\mathfrak{W}_{r}^{i} \textbf{t}$. As indicated by the interaction model, NTN defines
    for each relation a separate neural network which makes the model very expressive, but at the same time
    computationally expensive.

    .. seealso::

       - Original Implementation (Matlab): `<https://github.com/khurram18/NeuralTensorNetworks>`_
       - TensorFlow: `<https://github.com/dddoss/tensorflow-socher-ntn>`_
       - Keras: `<https://github.com/dapurv5/keras-neural-tensor-layer (Keras)>`_
    ---
    citation:
        author: Socher
        year: 2013
        link: https://dl.acm.org/doi/10.5555/2999611.2999715
        github: khurram18/NeuralTensorNetworks
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        num_slices=dict(type=int, low=2, high=4),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 100,
        num_slices: int = 4,
        non_linearity: Optional[nn.Module] = None,
        entity_initializer: Hint[Initializer] = None,
        **kwargs,
    ) -> None:
        r"""Initialize NTN.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 350]$.
        :param num_slices: The number of slices in the parameters
        :param non_linearity: A non-linear activation function. Defaults to the hyperbolic
            tangent :class:`torch.nn.Tanh`.
        :param entity_initializer: Entity initializer function. Defaults to :func:`torch.nn.init.uniform_`
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.EntityEmbeddingModel`
        """
        super().__init__(
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
            ),
            **kwargs,
        )
        self.num_slices = num_slices

        self.w = nn.Parameter(data=torch.empty(
            self.num_relations,
            num_slices,
            embedding_dim,
            embedding_dim,
            device=self.device,
        ), requires_grad=True)
        self.vh = nn.Parameter(data=torch.empty(
            self.num_relations,
            num_slices,
            embedding_dim,
            device=self.device,
        ), requires_grad=True)
        self.vt = nn.Parameter(data=torch.empty(
            self.num_relations,
            num_slices,
            embedding_dim,
            device=self.device,
        ), requires_grad=True)
        self.b = nn.Parameter(data=torch.empty(
            self.num_relations,
            num_slices,
            device=self.device,
        ), requires_grad=True)
        self.u = nn.Parameter(data=torch.empty(
            self.num_relations,
            num_slices,
            device=self.device,
        ), requires_grad=True)
        if non_linearity is None:
            non_linearity = nn.Tanh()
        self.non_linearity = non_linearity

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
        nn.init.normal_(self.w)
        nn.init.normal_(self.vh)
        nn.init.normal_(self.vt)
        nn.init.normal_(self.b)
        nn.init.normal_(self.u)

    def _score(
        self,
        h_indices: Optional[torch.LongTensor] = None,
        r_indices: Optional[torch.LongTensor] = None,
        t_indices: Optional[torch.LongTensor] = None,
        slice_size: int = None,
    ) -> torch.FloatTensor:
        """
        Compute scores for NTN.

        :param h_indices: shape: (batch_size,)
        :param r_indices: shape: (batch_size,)
        :param t_indices: shape: (batch_size,)
        :param slice_size: >0
            The divisor for the scoring function when using slicing.

        :return: shape: (batch_size, num_entities)
        """
        assert r_indices is not None

        #: shape: (batch_size, num_entities, d)
        h_all = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        t_all = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)

        if slice_size is None:
            return self._interaction_function(h=h_all, t=t_all, r_indices=r_indices)

        if h_all.shape[1] > t_all.shape[1]:
            h_was_split = True
            split_tensor = torch.split(h_all, slice_size, dim=1)
            constant_tensor = t_all
        else:
            h_was_split = False
            split_tensor = torch.split(t_all, slice_size, dim=1)
            constant_tensor = h_all

        scores_arr = []
        for split in split_tensor:
            if h_was_split:
                h = split
                t = constant_tensor
            else:
                h = constant_tensor
                t = split
            score = self._interaction_function(h=h, t=t, r_indices=r_indices)
            scores_arr.append(score)

        return torch.cat(scores_arr, dim=1)

    def _interaction_function(
        self,
        h: torch.FloatTensor,
        t: torch.FloatTensor,
        r_indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        #: Prepare h: (b, e, d) -> (b, e, 1, 1, d)
        h_for_w = h.unsqueeze(dim=-2).unsqueeze(dim=-2)

        #: Prepare t: (b, e, d) -> (b, e, 1, d, 1)
        t_for_w = t.unsqueeze(dim=-2).unsqueeze(dim=-1)

        #: Prepare w: (R, k, d, d) -> (b, k, d, d) -> (b, 1, k, d, d)
        w_r = self.w.index_select(dim=0, index=r_indices).unsqueeze(dim=1)

        # h.T @ W @ t, shape: (b, e, k, 1, 1)
        hwt = (h_for_w @ w_r @ t_for_w)

        #: reduce (b, e, k, 1, 1) -> (b, e, k)
        hwt = hwt.squeeze(dim=-1).squeeze(dim=-1)

        #: Prepare vh: (R, k, d) -> (b, k, d) -> (b, 1, k, d)
        vh_r = self.vh.index_select(dim=0, index=r_indices).unsqueeze(dim=1)

        #: Prepare h: (b, e, d) -> (b, e, d, 1)
        h_for_v = h.unsqueeze(dim=-1)

        # V_h @ h, shape: (b, e, k, 1)
        vhh = vh_r @ h_for_v

        #: reduce (b, e, k, 1) -> (b, e, k)
        vhh = vhh.squeeze(dim=-1)

        #: Prepare vt: (R, k, d) -> (b, k, d) -> (b, 1, k, d)
        vt_r = self.vt.index_select(dim=0, index=r_indices).unsqueeze(dim=1)

        #: Prepare t: (b, e, d) -> (b, e, d, 1)
        t_for_v = t.unsqueeze(dim=-1)

        # V_t @ t, shape: (b, e, k, 1)
        vtt = vt_r @ t_for_v

        #: reduce (b, e, k, 1) -> (b, e, k)
        vtt = vtt.squeeze(dim=-1)

        #: Prepare b: (R, k) -> (b, k) -> (b, 1, k)
        b = self.b.index_select(dim=0, index=r_indices).unsqueeze(dim=1)

        # a = f(h.T @ W @ t + Vh @ h + Vt @ t + b), shape: (b, e, k)
        pre_act = hwt + vhh + vtt + b
        act = self.non_linearity(pre_act)

        # prepare u: (R, k) -> (b, k) -> (b, 1, k, 1)
        u = self.u.index_select(dim=0, index=r_indices).unsqueeze(dim=1).unsqueeze(dim=-1)

        # prepare act: (b, e, k) -> (b, e, 1, k)
        act = act.unsqueeze(dim=-2)

        # compute score, shape: (b, e, 1, 1)
        score = act @ u

        # reduce
        score = score.squeeze(dim=-1).squeeze(dim=-1)

        return score

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_indices=hrt_batch[:, 0], r_indices=hrt_batch[:, 1], t_indices=hrt_batch[:, 2])

    def score_t(self, hr_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        return self._score(h_indices=hr_batch[:, 0], r_indices=hr_batch[:, 1], slice_size=slice_size)

    def score_h(self, rt_batch: torch.LongTensor, slice_size: int = None) -> torch.FloatTensor:  # noqa: D102
        return self._score(r_indices=rt_batch[:, 0], t_indices=rt_batch[:, 1], slice_size=slice_size)
