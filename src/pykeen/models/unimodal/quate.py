# -*- coding: utf-8 -*-

"""Implementation of the QuatE model."""
import math
from typing import Any, ClassVar, Mapping, Optional, Type

import torch
from torch import nn
from torch.nn import functional

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEWithLogitsLoss, Loss
from ...nn.emb import EmbeddingSpecification
from ...regularizers import LpRegularizer, Regularizer
from ...triples import TriplesFactory
from ...typing import DeviceHint, Hint, Initializer

__all__ = [
    'QuatE',
]

def init_quaternions(num_elements: int, dim: int) -> torch.FloatTensor:
    # scaling factor
    s = 1. / math.sqrt(2 * num_elements)
    # modulus ~ Uniform[-s, s]
    modulus = 2 * s * torch.rand(num_elements, dim) - s
    # phase ~ Uniform[0, 2*pi]
    phase = 2 * math.pi * torch.rand(num_elements, dim)
    # real part
    real = (modulus * phase.cos()).unsqueeze(dim=-1)
    # purely imaginary quaternions unitary
    imag = torch.rand(num_elements, dim, 3)
    imag = functional.normalize(imag, p=2, dim=-1)
    imag = imag * (modulus * phase.sin()).unsqueeze(dim=-1)
    x = torch.cat([real, imag], dim=-1)
    return x.view(num_elements, 4 * dim)


class QuatE(EntityRelationEmbeddingModel):
    r"""An implementation of QuatE [zhang2019]_.

    QuatE uses hypercomplex valued representations for the
    entities and relations. Entities and relations are represented as vectors
    $\textbf{e}_i, \textbf{r}_i \in \mathbb{H}^d$, and the plausibility score is computed using the
    quaternion inner product.

    .. seealso ::

        Official implementation: https://github.com/cheungdaven/QuatE/blob/master/models/QuatE.py
    ---
    citation:
        author: Zhang
        year: 2019
        link: https://arxiv.org/abs/1904.10281
        github: cheungdaven/quate
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = dict(reduction='mean')
    #: The regularizer used by [zhang2019]_ for QuatE.
    regularizer_default: ClassVar[Type[Regularizer]] = LpRegularizer
    #: The LP settings used by [zhang2019]_ for QuatE.
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.0025,
        p=3.0,
        normalize=True,
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 100,
        normalize_relations: bool = True,
        loss: Optional[Loss] = None,
        regularizer: Optional[Regularizer] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        entity_initializer: Hint[Initializer] = init_quaternions,
        relation_initializer: Hint[Initializer] = init_quaternions,
    ) -> None:
        """Initialize QuatE.

        :param triples_factory:
            The triple factory connected to the model.
        :param embedding_dim:
            The embedding dimensionality of the entity embeddings.
        :param loss:
            The loss to use. Defaults to SoftplusLoss.
        :param regularizer:
            The regularizer to use.
        :param preferred_device:
            The default device where to model is located.
        :param random_seed:
            An optional random seed to set before the initialization of weights.
        """
        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            entity_representations=EmbeddingSpecification(
                embedding_dim=4 * embedding_dim,
                initializer=entity_initializer,
                dtype=torch.float,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=4 * embedding_dim,
                initializer=relation_initializer,
                dtype=torch.float,
            ),
        )
        self.normalize_relations = normalize_relations
        self.real_embedding_dim = embedding_dim


    def init_empty_weights_(self):  # noqa: D102

        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            _weight=init_quaternions(
                num_elements=self.num_entities,
                dim=self.real_embedding_dim,
            ),
        )

        self.relation_embeddings = nn.Embedding(
            self.num_relations,
            self.embedding_dim,
            _weight=init_quaternions(
                num_elements=self.num_relations,
                dim=self.real_embedding_dim,
            ),
        )

        return self

    def post_parameter_update(self) -> None:  # noqa: D102
        r"""Normalize the length of relation vectors, if the forward constraint has not been applied yet.

        Absolute value of complex number

        .. math::

            |a + bi + cj dk| = \sqrt{a^2 + b^2 + c^2 + d^2}

        L2 norm of quaternion vector:

        .. math::
            \|x\|^2 = \sum_{i=1}^d |x_i|^2
                     = \sum_{i=1}^d (x_i.re^2 + x_i.im_1^2 + x_i.im_2^2 + x_i.im_3^2)
        """
        # Make sure to call super first
        super().post_parameter_update()

        if self.normalize_relations:
            # Normalize relation embeddings
            rel = self.relation_embeddings.weight.data.view(self.num_relations, self.real_embedding_dim, 4)
            rel = functional.normalize(rel, p=2, dim=-1)
            self.relation_embeddings.weight.data = rel.view(self.num_relations, self.embedding_dim)

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function of QuatE for given embeddings.

        The embeddings have to be in a broadcastable shape.

        WARNING: No forward constraints are applied.

        :param h: shape: (..., e, 4)
            Head embeddings. Last dimension corresponds to (real, imag, imag,imag).
        :param r: shape: (..., e, 4)
            Relation embeddings. Last dimension corresponds to (real, imag, imag,imag).
        :param t: shape: (..., e, 4)
            Tail embeddings. Last dimension corresponds to (real, imag, imag,imag).

        :return: shape: (...)
            The scores.
        """
        # Decompose into real and imaginary part
        h_a = h[..., 0]
        h_b = h[..., 1]
        h_c = h[..., 2]
        h_d = h[..., 3]
        r_a = r[..., 0]
        r_b = r[..., 1]
        r_c = r[..., 2]
        r_d = r[..., 3]

        # Rotate (=Hamilton product in quaternion space).
        rot_h = torch.stack(
            [
                h_a * r_a - h_b * r_b - h_c * r_c - h_d * r_d,
                h_a * r_b + h_b * r_a + h_c * r_d - h_d * r_c,
                h_a * r_c - h_b * r_d + h_c * r_a + h_d * r_b,
                h_a * r_d + h_b * r_c - h_c * r_b + h_d * r_a,
            ],
            dim=-1,
        )

        inner_prod = rot_h * t
        scores = -inner_prod.sum(dim=[-2, -1])

        return scores

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:
        # Get embeddings
        h = self.entity_embeddings(hr_batch[:, 0]).view(-1, 1, self.real_embedding_dim, 4)
        r = self.relation_embeddings(hr_batch[:, 1]).view(-1, 1, self.real_embedding_dim, 4)

        # Rank against all entities
        t = self.entity_embeddings.weight.view(1, -1, self.real_embedding_dim, 4)

        # Compute scores
        scores = self.interaction_function(h=h, r=r, t=t)

        # Embedding Regularization
        self.regularize_if_necessary(h.view(-1, self.embedding_dim), t.view(-1, self.embedding_dim))

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        r = self.relation_embeddings(rt_batch[:, 0]).view(-1, 1, self.real_embedding_dim, 4)
        t = self.entity_embeddings(rt_batch[:, 1]).view(-1, 1, self.real_embedding_dim, 4)

        # Conjugation is an involution and is its own inverse (see https://arxiv.org/pdf/1904.10281.pdf)
        r_inv = torch.stack([r[:, :, :, 0], -r[:, :, :, 1], -r[:, :, :, 2], -r[:, :, :, 3]], dim=-1)

        # Rank against all entities
        h = self.entity_embeddings.weight.view(1, -1, self.real_embedding_dim, 4)

        # Compute scores
        # Q_h ⊗ W_r·Q_t = Q_t ⊗ ̄W_r·Q_h (see https://arxiv.org/pdf/1904.10281.pdf)
        scores = self.interaction_function(h=t, r=r_inv, t=h)

        # Embedding Regularization
        self.regularize_if_necessary(h.view(-1, self.embedding_dim), t.view(-1, self.embedding_dim))

        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(hrt_batch[:, 0]).view(-1, self.real_embedding_dim, 4)
        r = self.relation_embeddings(hrt_batch[:, 1]).view(-1, self.real_embedding_dim, 4)
        t = self.entity_embeddings(hrt_batch[:, 2]).view(-1, self.real_embedding_dim, 4)

        # Compute scores
        scores = self.interaction_function(h=h, r=r, t=t).view(-1, 1)

        # Embedding Regularization
        self.regularize_if_necessary(h.view(-1, self.embedding_dim), t.view(-1, self.embedding_dim))

        return scores