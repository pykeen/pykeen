# coding=utf-8
"""Implementation of the HAKE model."""

from typing import Optional

import torch
from torch import nn

from pykeen.losses import Loss
from pykeen.models import EntityRelationEmbeddingModel
from pykeen.regularizers import Regularizer
from pykeen.triples import TriplesFactory
from pykeen.utils import get_embedding, get_embedding_in_long_canonical_shape


class HAKE(EntityRelationEmbeddingModel):
    """An implementation of HAKE [zhang2020]_.

    .. seealso ::
        Official implementation: https://github.com/MIRALab-USTC/KGE-HAKE

    """

    # There are no value ranges for HPO given in the paper
    # These values are the values ranges used in the usage examples from
    # https://github.com/MIRALab-USTC/KGE-HAKE/blob/31049c2a00492533693699636d9c0a3a2b299fc8/README.md
    hpo_default = dict(
        embedding_dim=dict(type=int, low=256, high=512, q=32),
        gamma=dict(type=float, low=6.0, high=24.0),
        modulus_weight=dict(type=float, low=0.5, high=4.0),
        phase_weight=dict(type=float, low=0.5, high=1.0),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 500,
        gamma: float = 12.0,
        epsilon: float = 2.0,
        modulus_weight: float = 1.0,
        phase_weight: float = 0.5,
        loss: Optional[Loss] = None,
        predict_with_sigmoid: bool = False,
        automatic_memory_optimization: Optional[bool] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ):
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            relation_dim=embedding_dim,
            loss=loss,
            predict_with_sigmoid=predict_with_sigmoid,
            automatic_memory_optimization=automatic_memory_optimization,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        # entity / relation modulus
        self.entity_modulus, self.relation_modulus = [
            get_embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                device=self.device,
            )
            for num_embeddings in (self.num_entities, self.num_relations)
        ]

        # relation bias
        self.relation_bias = get_embedding(
            num_embeddings=self.num_relations,
            embedding_dim=embedding_dim,
            device=self.device,
        )

        self.gamma = gamma
        self.embedding_range = (gamma + epsilon) / embedding_dim
        self.initial_phase_weight = phase_weight * self.embedding_range
        self.initial_modulus_weight = modulus_weight

        # trainable phase / modulus weights
        self.phase_weight = nn.Parameter(data=torch.empty(1))
        self.modulus_weight = nn.Parameter(data=torch.empty(1))

        # Initialize parameters
        self.reset_parameters_()

    def _reset_parameters_(self):  # noqa: D102
        # Initialize embeddings
        for embeddings in (
            self.entity_embeddings,
            self.relation_embeddings,
        ):
            nn.init.uniform_(
                tensor=embeddings.weight,
                a=-self.embedding_range,
                b=self.embedding_range,
            )

        # Initialize modulus: uniform for entity, one for relation
        nn.init.uniform_(
            tensor=self.entity_modulus.weight,
            a=-self.embedding_range,
            b=self.embedding_range,
        )
        nn.init.ones_(tensor=self.relation_modulus.weight)

        # Initialize relation bias
        nn.init.zeros_(tensor=self.relation_bias.weight)

        # Phase / Modulus weight
        nn.init.constant_(tensor=self.phase_weight, val=self.initial_phase_weight)
        nn.init.constant_(tensor=self.modulus_weight, val=self.initial_modulus_weight)

    def _score(
        self,
        h_ind: Optional[torch.LongTensor],
        r_ind: Optional[torch.LongTensor],
        t_ind: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:
        # Lookup embeddings
        phase_h, mod_h, phase_r, mod_r, bias_r, phase_t, mod_t = [
            get_embedding_in_long_canonical_shape(embedding=embedding, ind=ind, col=col)
            for ind, col, embedding in (
                (h_ind, 1, self.entity_embeddings),
                (h_ind, 1, self.entity_modulus),
                (r_ind, 2, self.relation_embeddings),
                (r_ind, 2, self.relation_modulus),
                (r_ind, 2, self.relation_bias),
                (t_ind, 3, self.entity_embeddings),
                (t_ind, 3, self.entity_modulus),
            )
        ]

        # ???
        bias_r = bias_r.clamp(max=1)
        mod_r = mod_r.abs()
        indicator = (bias_r < -mod_r)
        bias_r[indicator] = -mod_r[indicator]

        # compute phase score
        phase_score = phase_h + phase_r - phase_t
        phase_score = (0.5 * phase_score).sin().norm(p=1, dim=-1)

        # compute modulus score
        modulus_score = mod_h * (mod_r + bias_r) - mod_t * (1 - bias_r)
        modulus_score = modulus_score.norm(p=2, dim=-1)

        # combine
        return self.gamma - (self.phase_weight * phase_score + self.modulus_weight * modulus_score)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(
            h_ind=hrt_batch[:, 0],
            r_ind=hrt_batch[:, 1],
            t_ind=hrt_batch[:, 2],
        ).view(hrt_batch.shape[0], 1)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(
            h_ind=None,
            r_ind=rt_batch[:, 0],
            t_ind=rt_batch[:, 1],
        ).view(rt_batch.shape[0], self.num_entities)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self._score(
            h_ind=hr_batch[:, 0],
            r_ind=hr_batch[:, 1],
            t_ind=None,
        ).view(hr_batch.shape[0], self.num_entities)
