# -*- coding: utf-8 -*-

"""Implementation of the RotatE model."""

from typing import Any, ClassVar, Mapping

import torch
import torch.autograd

from ..base import EntityRelationEmbeddingModel
from ...nn.emb import EmbeddingSpecification
from ...nn.init import init_phases, xavier_uniform_
from ...typing import Constrainer, Hint, Initializer
from ...utils import complex_normalize

__all__ = [
    'RotatE',
]


class RotatE(EntityRelationEmbeddingModel):
    r"""An implementation of RotatE from [sun2019]_.

    RotatE models relations as rotations from head to tail entities in complex space:

    .. math::

        \textbf{e}_t= \textbf{e}_h \odot \textbf{r}_r

    where $\textbf{e}, \textbf{r} \in \mathbb{C}^{d}$ and the complex elements of
    $\textbf{r}_r$ are restricted to have a modulus of one ($\|\textbf{r}_r\| = 1$). The
    interaction model is then defined as:

    .. math::

        f(h,r,t) = -\|\textbf{e}_h \odot \textbf{r}_r - \textbf{e}_t\|

    which allows to model symmetry, antisymmetry, inversion, and composition.

    .. seealso::

       - Authors' `implementation of RotatE
         <https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/model.py#L200-L228>`_
    ---
    citation:
        author: Sun
        year: 2019
        link: https://arxiv.org/abs/1902.10197v1
        github: DeepGraphLearning/KnowledgeGraphEmbedding
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=dict(type=int, low=32, high=1024, q=16),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = init_phases,
        relation_constrainer: Hint[Constrainer] = complex_normalize,
        **kwargs,
    ) -> None:
        super().__init__(
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
                dtype=torch.cfloat,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
                dtype=torch.cfloat,
            ),
            **kwargs,
        )
        self.real_embedding_dim = embedding_dim

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function of ComplEx for given embeddings.

        The embeddings have to be in a broadcastable shape.

        WARNING: No forward constraints are applied.

        :param h: shape: (..., e, 2)
            Head embeddings. Last dimension corresponds to (real, imag).
        :param r: shape: (..., e, 2)
            Relation embeddings. Last dimension corresponds to (real, imag).
        :param t: shape: (..., e, 2)
            Tail embeddings. Last dimension corresponds to (real, imag).

        :return: shape: (...)
            The scores.
        """
        # Decompose into real and imaginary part
        h_re = h[..., 0]
        h_im = h[..., 1]
        r_re = r[..., 0]
        r_im = r[..., 1]

        # Rotate (=Hadamard product in complex space).
        rot_h = torch.stack(
            [
                h_re * r_re - h_im * r_im,
                h_re * r_im + h_im * r_re,
            ],
            dim=-1,
        )
        # Workaround until https://github.com/pytorch/pytorch/issues/30704 is fixed
        diff = rot_h - t
        scores = -torch.norm(diff.view(diff.shape[:-2] + (-1,)), dim=-1)

        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0]).view(-1, self.real_embedding_dim, 2)
        r = self.relation_embeddings(indices=hrt_batch[:, 1]).view(-1, self.real_embedding_dim, 2)
        t = self.entity_embeddings(indices=hrt_batch[:, 2]).view(-1, self.real_embedding_dim, 2)

        # Compute scores
        scores = self.interaction_function(h=h, r=r, t=t).view(-1, 1)

        # Embedding Regularization
        self.regularize_if_necessary(h.view(-1, self.embedding_dim), t.view(-1, self.embedding_dim))

        return scores

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0]).view(-1, 1, self.real_embedding_dim, 2)
        r = self.relation_embeddings(indices=hr_batch[:, 1]).view(-1, 1, self.real_embedding_dim, 2)

        # Rank against all entities
        t = self.entity_embeddings(indices=None).view(1, -1, self.real_embedding_dim, 2)

        # Compute scores
        scores = self.interaction_function(h=h, r=r, t=t)

        # Embedding Regularization
        self.regularize_if_necessary(h.view(-1, self.embedding_dim), t.view(-1, self.embedding_dim))

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        r = self.relation_embeddings(indices=rt_batch[:, 0]).view(-1, 1, self.real_embedding_dim, 2)
        t = self.entity_embeddings(indices=rt_batch[:, 1]).view(-1, 1, self.real_embedding_dim, 2)

        # r expresses a rotation in complex plane.
        # The inverse rotation is expressed by the complex conjugate of r.
        # The score is computed as the distance of the relation-rotated head to the tail.
        # Equivalently, we can rotate the tail by the inverse relation, and measure the distance to the head, i.e.
        # |h * r - t| = |h - conj(r) * t|
        r_inv = torch.stack([r[:, :, :, 0], -r[:, :, :, 1]], dim=-1)

        # Rank against all entities
        h = self.entity_embeddings(indices=None).view(1, -1, self.real_embedding_dim, 2)

        # Compute scores
        scores = self.interaction_function(h=t, r=r_inv, t=h)

        # Embedding Regularization
        self.regularize_if_necessary(h.view(-1, self.embedding_dim), t.view(-1, self.embedding_dim))

        return scores
