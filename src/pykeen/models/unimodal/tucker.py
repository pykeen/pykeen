# -*- coding: utf-8 -*-

"""Implementation of TuckEr."""

from typing import Any, ClassVar, Mapping, Optional, Type

import torch
import torch.autograd
from torch import nn

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn.emb import EmbeddingSpecification
from ...nn.init import xavier_normal_
from ...typing import Hint, Initializer

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
    r"""An implementation of TuckEr from [balazevic2019]_.

    TuckER is a linear model that is based on the tensor factorization method Tucker in which a three-mode tensor
    $\mathfrak{X} \in \mathbb{R}^{I \times J \times K}$ is decomposed into a set of factor matrices
    $\textbf{A} \in \mathbb{R}^{I \times P}$, $\textbf{B} \in \mathbb{R}^{J \times Q}$, and
    $\textbf{C} \in \mathbb{R}^{K \times R}$ and a core tensor
    $\mathfrak{Z} \in \mathbb{R}^{P \times Q \times R}$ (of lower rank):

    .. math::

        \mathfrak{X} \approx \mathfrak{Z} \times_1 \textbf{A} \times_2 \textbf{B} \times_3 \textbf{C}

    where $\times_n$ is the tensor product, with $n$ denoting along which mode the tensor product is computed.
    In TuckER, a knowledge graph is considered as a binary tensor which is factorized using the Tucker factorization
    where $\textbf{E} = \textbf{A} = \textbf{C} \in \mathbb{R}^{n_{e} \times d_e}$ denotes the entity embedding
    matrix, $\textbf{R} = \textbf{B} \in \mathbb{R}^{n_{r} \times d_r}$ represents the relation embedding matrix,
    and $\mathfrak{W} = \mathfrak{Z} \in \mathbb{R}^{d_e \times d_r \times d_e}$ is the *core tensor* that
    indicates the extent of interaction between the different factors. The interaction model is defined as:

    .. math::

        f(h,r,t) = \mathfrak{W} \times_1 \textbf{h} \times_2 \textbf{r} \times_3 \textbf{t}

    where $\textbf{h},\textbf{t}$ correspond to rows of $\textbf{E}$ and $\textbf{r}$ to a row of $\textbf{R}$.

    The dropout values correspond to the following dropouts in the model's score function:

    .. math::

        \text{Dropout}_2(BN(\text{Dropout}_0(BN(h)) \times_1 \text{Dropout}_1(W \times_2 r))) \times_3 t

    where h,r,t are the head, relation, and tail embedding, W is the core tensor, \times_i denotes the tensor
    product along the i-th mode, BN denotes batch normalization, and :math:`\text{Dropout}` dropout.

    .. seealso::

       - Official implementation: https://github.com/ibalazevic/TuckER
       - pykg2vec implementation of TuckEr https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TuckER.py
    ---
    citation:
        author: Balažević
        year: 2019
        link: https://arxiv.org/abs/1901.09590
        github: ibalazevic/TuckER
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        relation_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        dropout_0=DEFAULT_DROPOUT_HPO_RANGE,
        dropout_1=DEFAULT_DROPOUT_HPO_RANGE,
        dropout_2=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        relation_dim: Optional[int] = None,
        dropout_0: float = 0.3,
        dropout_1: float = 0.4,
        dropout_2: float = 0.5,
        apply_batch_normalization: bool = True,
        entity_initializer: Hint[Initializer] = xavier_normal_,
        relation_initializer: Hint[Initializer] = xavier_normal_,
        **kwargs,
    ) -> None:
        super().__init__(
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=relation_dim or embedding_dim,
                initializer=relation_initializer,
            ),
            **kwargs,
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

    def _reset_parameters_(self):  # noqa: D102
        super()._reset_parameters_()
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
        h = self.entity_embeddings(indices=hrt_batch[:, 0]).unsqueeze(1)
        r = self.relation_embeddings(indices=hrt_batch[:, 1])
        t = self.entity_embeddings(indices=hrt_batch[:, 2]).unsqueeze(1)

        # Compute scores
        scores = self._scoring_function(h=h, r=r, t=t)

        return scores

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0]).unsqueeze(1)
        r = self.relation_embeddings(indices=hr_batch[:, 1])
        t = self.entity_embeddings(indices=None).unsqueeze(0)

        # Compute scores
        scores = self._scoring_function(h=h, r=r, t=t)

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=None).unsqueeze(0)
        r = self.relation_embeddings(indices=rt_batch[:, 0])
        t = self.entity_embeddings(indices=rt_batch[:, 1]).unsqueeze(1)

        # Compute scores
        scores = self._scoring_function(h=h, r=r, t=t)

        return scores
