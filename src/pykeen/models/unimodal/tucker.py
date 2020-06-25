# -*- coding: utf-8 -*-

"""Implementation of TuckEr."""

from typing import Optional

import torch
import torch.autograd
from torch import nn

from ..base import EntityRelationEmbeddingModel
from ..init import embedding_xavier_normal_
from ...losses import BCEAfterSigmoidLoss, Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory

__all__ = [
    'TuckER',
]


def _apply_bn_to_tensor(
    batch_norm: nn.BatchNorm1d,
    tensor: torch.FloatTensor,
) -> torch.FloatTensor:
    shape = tensor.shape
    tensor = tensor.view(-1, shape[-1])
    tensor = batch_norm(tensor)
    tensor = tensor.view(*shape)
    return tensor


class TuckER(EntityRelationEmbeddingModel):
    """An implementation of TuckEr from [balazevic2019]_.

    This model uses the Tucker tensor factorization.

    .. seealso::

       - Official implementation: <https://github.com/ibalazevic/TuckER>
       - pykg2vec implementation of TuckEr  <https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TuckER.py>
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        relation_dim=dict(type=int, low=30, high=200, q=25),
        dropout_0=dict(type=float, low=0.1, high=0.4),
        dropout_1=dict(type=float, low=0.1, high=0.5),
        dropout_2=dict(type=float, low=0.1, high=0.6),
    )
    #: The default loss function class
    loss_default = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = {}

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 200,
        automatic_memory_optimization: Optional[bool] = None,
        relation_dim: Optional[int] = None,
        loss: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        dropout_0: float = 0.3,
        dropout_1: float = 0.4,
        dropout_2: float = 0.5,
        regularizer: Optional[Regularizer] = None,
        apply_batch_normalization: bool = True,
    ) -> None:
        """Initialize the model.

        The dropout values correspond to the following dropouts in the model's score function:

            DO_2(BN(DO_0(BN(h)) x_1 DO_1(W x_2 r))) x_3 t

        where h,r,t are the head, relation, and tail embedding, W is the core tensor, x_i denotes the tensor
        product along the i-th mode, BN denotes batch normalization, and DO dropout.
        """
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            relation_dim=relation_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        # Core tensor
        # Note: we use a different dimension permutation as in the official implementation to match the paper.
        self.core_tensor = nn.Parameter(
            torch.empty(self.embedding_dim, self.relation_dim, self.embedding_dim, device=self.device),
            requires_grad=True,
        )

        # Dropout
        self.input_dropout = nn.Dropout(dropout_0)
        self.hidden_dropout_1 = nn.Dropout(dropout_1)
        self.hidden_dropout_2 = nn.Dropout(dropout_2)

        self.apply_batch_normalization = apply_batch_normalization

        if self.apply_batch_normalization:
            self.bn_0 = nn.BatchNorm1d(self.embedding_dim)
            self.bn_1 = nn.BatchNorm1d(self.embedding_dim)

        # Finalize initialization
        self.reset_parameters_()

    def _reset_parameters_(self):  # noqa: D102
        embedding_xavier_normal_(self.entity_embeddings)
        embedding_xavier_normal_(self.relation_embeddings)
        # Initialize core tensor, cf. https://github.com/ibalazevic/TuckER/blob/master/model.py#L12
        nn.init.uniform_(self.core_tensor, -1., 1.)

    def _scoring_function(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Evaluate the scoring function.

        Compute scoring function W x_1 h x_2 r x_3 t as in the official implementation, i.e. as

            DO(BN(DO(BN(h)) x_1 DO(W x_2 r))) x_3 t

        where BN denotes BatchNorm and DO denotes Dropout

        :param h: shape: (batch_size, 1, embedding_dim) or (1, num_entities, embedding_dim)
        :param r: shape: (batch_size, relation_dim)
        :param t: shape: (1, num_entities, embedding_dim) or (batch_size, 1, embedding_dim)
        :return: shape: (batch_size, num_entities) or (batch_size, 1)
        """
        # Abbreviation
        w = self.core_tensor
        d_e = self.embedding_dim
        d_r = self.relation_dim

        # Compute h_n = DO(BN(h))
        if self.apply_batch_normalization:
            h = _apply_bn_to_tensor(batch_norm=self.bn_0, tensor=h)

        h = self.input_dropout(h)

        # Compute wr = DO(W x_2 r)
        w = w.view(1, d_e, d_r, d_e)
        r = r.view(-1, 1, 1, d_r)
        wr = r @ w
        wr = self.hidden_dropout_1(wr)

        # compute whr = DO(BN(h_n x_1 wr))
        wr = wr.view(-1, d_e, d_e)
        whr = (h @ wr)
        if self.apply_batch_normalization:
            whr = _apply_bn_to_tensor(batch_norm=self.bn_1, tensor=whr)
        whr = self.hidden_dropout_2(whr)

        # Compute whr x_3 t
        scores = torch.sum(whr * t, dim=-1)

        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hrt_batch[:, 0]).unsqueeze(1)
        r = self.relation_embeddings(hrt_batch[:, 1])
        t = self.entity_embeddings(hrt_batch[:, 2]).unsqueeze(1)

        # Compute scores
        scores = self._scoring_function(h=h, r=r, t=t)

        return scores

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hr_batch[:, 0]).unsqueeze(1)
        r = self.relation_embeddings(hr_batch[:, 1])
        t = self.entity_embeddings.weight.unsqueeze(0)

        # Compute scores
        scores = self._scoring_function(h=h, r=r, t=t)

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings.weight.unsqueeze(0)
        r = self.relation_embeddings(rt_batch[:, 0])
        t = self.entity_embeddings(rt_batch[:, 1]).unsqueeze(1)

        # Compute scores
        scores = self._scoring_function(h=h, r=r, t=t)

        return scores
