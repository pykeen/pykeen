# -*- coding: utf-8 -*-

"""Tests for literal interaction functions."""
from typing import Any, MutableMapping

import torch
from torch import nn

from pykeen.datasets.nations import NationsLiteral
from pykeen.nn.modules import ComplExInteraction, LiteralInteraction
from pykeen.triples.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from tests import cases


class LiteralTests(cases.InteractionTestCase):
    """Tests for LiteralInteraction function."""

    cls = LiteralInteraction
    kwargs = dict(
        base=ComplExInteraction(),
    )

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        # TODO this doesnt make sense
        triples_factory: TriplesNumericLiteralsFactory = NationsLiteral().training
        extra_dim = triples_factory.numeric_literals.shape[1]
        kwargs["combination"] = nn.Sequential(
            nn.Linear(self.dim + extra_dim, self.dim),
            nn.Dropout(0.1),
        )
        self.shape_kwargs["e"] = extra_dim
        return kwargs

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        return self.instance.base(
            self.instance.combination(torch.cat(h, dim=-1)),
            r,
            self.instance.combination(torch.cat(t, dim=-1)),
        )
