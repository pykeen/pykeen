# -*- coding: utf-8 -*-

"""Implementation of the ComplexLiteral model based on the local closed world assumption (LCWA) training approach."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from ..unimodal.complex import ComplEx
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEWithLogitsLoss, Loss
from ...nn import Embedding
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint
# TODO: Check entire build of the model
from ...utils import split_complex


class ComplExLiteral(ComplEx):
    """An implementation of ComplexLiteral from [agustinus2018]_ based on the LCWA training approach."""

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        input_dropout=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default = BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs = {}

    def __init__(
        self,
        triples_factory: TriplesNumericLiteralsFactory,
        embedding_dim: int = 50,
        input_dropout: float = 0.2,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            entity_initializer=xavier_normal_,
            relation_initializer=xavier_normal_,
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

        self.real_non_lin_transf = torch.nn.Sequential(
            nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim),
            torch.nn.Tanh(),
        )

        self.img_non_lin_transf = torch.nn.Sequential(
            nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim),
            torch.nn.Tanh(),
        )

        self.inp_drop = torch.nn.Dropout(input_dropout)

    def _apply_g_function(
        self,
        emb: torch.FloatTensor,
        lit: torch.FloatTensor,
        dropout: bool,
    ):
        if dropout:
            emb = self.inp_drop(emb)
        re, im = split_complex(emb)
        re, im = [torch.cat([x, lit], dim=-1) for x in (re, im)]
        re, im = [
            trans(x.view(-1, x.shape[-1])).view(*x.shape)
            for x, trans in (
                (re, self.real_non_lin_transf),
                (im, self.img_non_lin_transf),
            )
        ]
        return torch.cat([re, im], dim=-1)

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:
        """Unified score function."""
        # get embeddings
        h = self.entity_embeddings.get_in_canonical_shape(indices=h_indices)
        r = self.relation_embeddings.get_in_canonical_shape(indices=r_indices)
        t = self.entity_embeddings.get_in_canonical_shape(indices=t_indices)

        # get literals
        h_lit, t_lit = [self.numeric_literals.get_in_canonical_shape(indices=i) for i in (h_indices, t_indices)]

        # combine
        h, t = [
            self._apply_g_function(emb, lit, dropout=dropout) for emb, lit, dropout in (
                (h, h_lit, True),
                (t, t_lit, False),
            )
        ]

        # dropout
        h, r = [self.inp_drop(x) for x in (h, r)]

        # Compute scores
        return self.interaction_function(h=h, r=r, t=t)
