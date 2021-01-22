# -*- coding: utf-8 -*-

"""Tests for literal interaction functions."""

import torch
from torch import nn

from pykeen.datasets.nations import NationsLiteral
from pykeen.nn.modules import ComplExInteraction, LiteralInteraction
from pykeen.triples.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from tests import cases

nations_literal = NationsLiteral()


class MockLiteralInteraction(LiteralInteraction):
    """A mock literal interaction around ComplEx."""

    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        base = ComplExInteraction()

        # TODO this doesnt make sense
        triples_factory: TriplesNumericLiteralsFactory = nations_literal.training
        combination = nn.Sequential(
            nn.Linear(embedding_dim + triples_factory.numeric_literals.shape[1], embedding_dim),
            nn.Dropout(dropout),
        )
        super().__init__(
            base=base,
            combination=combination,
        )


class LiteralTests(cases.InteractionTestCase):
    """Tests for LiteralInteraction function."""

    cls = MockLiteralInteraction
    kwargs = {
        'embedding_dim': cases.InteractionTestCase.dim,
    }

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        return self.instance.base(
            self.instance.combination(torch.cat(h, dim=-1)),
            r,
            self.instance.combination(torch.cat(t, dim=-1)),
        )
