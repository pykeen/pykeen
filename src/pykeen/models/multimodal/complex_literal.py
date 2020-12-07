# -*- coding: utf-8 -*-

"""Implementation of the ComplexLiteral model based on the local closed world assumption (LCWA) training approach."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from ..base import MultimodalModel
from ..unimodal.complex import ComplEx
from ...constants import DEFAULT_DROPOUT_HPO_RANGE, DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import BCEWithLogitsLoss, Loss
from ...triples import TriplesNumericLiteralsFactory
from ...typing import DeviceHint


# TODO: Check entire build of the model
class ComplExLiteral(MultimodalModel):
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
        )

        self.entity_embs_real = None
        self.entity_embs_img = None
        self.relation_embs_real = None
        self.relation_embs_img = None

        # Literal
        # num_ent x num_lit
        numeric_literals = triples_factory.numeric_literals
        self.numeric_literals = nn.Embedding.from_pretrained(
            torch.tensor(numeric_literals, dtype=torch.float, device=self.device), freeze=True,
        )
        # Number of columns corresponds to number of literals
        self.num_of_literals = self.numeric_literals.weight.data.shape[1]

        self.real_non_lin_transf = torch.nn.Sequential(
            nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim),
            torch.nn.Tanh(),
        )

        self.img_non_lin_transf = torch.nn.Sequential(
            nn.Linear(self.embedding_dim + self.num_of_literals, self.embedding_dim),
            torch.nn.Tanh(),
        )

        self.inp_drop = torch.nn.Dropout(input_dropout)

        self.entity_embs_real = nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.entity_embs_img = nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.relation_embs_real = nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.relation_embs_img = nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)

    def _reset_parameters_(self):
        xavier_normal_(self.entity_embs_real.weight.data)
        xavier_normal_(self.entity_embs_img.weight.data)
        xavier_normal_(self.relation_embs_real.weight.data)
        xavier_normal_(self.relation_embs_img.weight.data)

    def _apply_g_function(
        self,
        idx: torch.LongTensor,
        dropout: bool,
    ):
        re = self.entity_embs_real(idx).view(-1, self.embedding_dim)
        im = self.entity_embs_img(idx).view(-1, self.embedding_dim)
        lit = self.numeric_literals(idx).view(-1, self.num_of_literals)
        if dropout:
            re, im = [self.inp_drop(x) for x in (re, im)]
        real = self.real_non_lin_transf(torch.cat([re, lit], 1))
        img = self.img_non_lin_transf(torch.cat([im, lit], 1))
        return real, img

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa:D102
        hi, ri, ti = hrt_batch.t()

        # get entity representations combined with literals
        h_re, h_im = self._apply_g_function(idx=hi, dropout=True)
        t_re, t_im = self._apply_g_function(idx=ti, dropout=True)

        # get relation representations
        r_re = self.inp_drop(self.relation_embs_real(ri)).view(-1, self.embedding_dim)
        r_im = self.inp_drop(self.relation_embs_img(ri)).view(-1, self.embedding_dim)

        # dropout for h + r
        h_re, h_im, r_re, r_im = [self.inp_drop(x) for x in (h_re, h_im, r_re, r_im)]

        h, r, t = [torch.cat([re, im], dim=-1) for re, im in ((h_re, h_im), (r_re, r_im), (t_re, t_im))]
        return ComplEx.interaction_function(h=h, r=r, t=t)
