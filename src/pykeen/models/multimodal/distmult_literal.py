# -*- coding: utf-8 -*-

"""Implementation of the DistMultLiteral model."""

from typing import Optional

import torch
import torch.nn as nn

from ..base import MultimodalModel
from ..unimodal.distmult import DistMult
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss
from ...nn import Embedding
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint


class DistMultLiteral(DistMult, MultimodalModel):
    """An implementation of DistMultLiteral from [agustinus2018]_."""

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        input_dropout=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default parameters for the default loss function class
    loss_default_kwargs = dict(margin=0.0)

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        input_dropout: float = 0.0,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
        )

        # Literal
        # num_ent x num_lit
        self.numeric_literals = Embedding(
            num_embeddings=triples_factory.num_entities,
            embedding_dim=triples_factory.numeric_literals.shape[-1],
            initializer=lambda x: triples_factory.numeric_literals,
        )
        # Number of columns corresponds to number of literals
        self.num_of_literals = self.numeric_literals.embedding_dim
        self.linear_transformation = nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim)
        self.inp_drop = torch.nn.Dropout(input_dropout)

    def _get_entity_representations(
        self,
        idx: torch.LongTensor,
    ) -> torch.FloatTensor:
        emb = self.entity_embeddings.get_in_canonical_shape(indices=idx)
        lit = self.numeric_literals.get_in_canonical_shape(indices=idx)
        x = self.linear_transformation(torch.cat([emb, lit], dim=-1))
        return self.inp_drop(x)

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:  # noqa: D102
        # TODO: this is very similar to ComplExLiteral, except a few dropout differences
        h = self._get_entity_representations(idx=h_indices)
        r = self.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        t = self._get_entity_representations(idx=t_indices)
        return self.interaction_function(h=h, r=r, t=t)

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self(h_indices=hrt_batch[:, 0], r_indices=hrt_batch[:, 1], t_indices=hrt_batch[:, 2]).view(-1, 1)

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self(h_indices=hr_batch[:, 0], r_indices=hr_batch[:, 1], t_indices=None)

    def score_r(self, ht_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self(h_indices=ht_batch[:, 0], r_indices=None, t_indices=ht_batch[:, 1])

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return self(h_indices=None, r_indices=rt_batch[:, 0], t_indices=rt_batch[:, 1])
