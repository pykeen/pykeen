# -*- coding: utf-8 -*-

"""Tests for literal interaction functions."""

from typing import Any, MutableMapping, Tuple

import torch

from pykeen.datasets.nations import NationsLiteral
from pykeen.nn.combinations import ComplExLiteralCombination, DistMultCombination
from pykeen.nn.modules import ComplExInteraction, DistMultInteraction
from pykeen.triples.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from tests import cases


class DistMultLiteralTestCase(cases.LiteralTestCase):
    """Tests for LiteralInteraction function."""

    kwargs = dict(
        base=DistMultInteraction(),
    )

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        triples_factory: TriplesNumericLiteralsFactory = NationsLiteral().training
        literal_embedding_dim = triples_factory.numeric_literals.shape[1]
        kwargs["combination"] = DistMultCombination(
            entity_embedding_dim=self.dim,
            literal_embedding_dim=literal_embedding_dim,
            input_dropout=0.1,
        )
        self.shape_kwargs["e"] = literal_embedding_dim
        return kwargs


class ComplExLiteralTestCase(cases.LiteralTestCase):
    """Tests for LiteralInteraction function."""

    kwargs = dict(
        base=ComplExInteraction(),
    )
    dtype = torch.cfloat

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        triples_factory: TriplesNumericLiteralsFactory = NationsLiteral().training
        literal_embedding_dim = triples_factory.numeric_literals.shape[1]
        kwargs["combination"] = ComplExLiteralCombination(
            # typically, the model takes care of adjusting the dimension size for "complex"
            # tensors, but we have to do it manually here for testing purposes
            entity_embedding_dim=self.dim,
            literal_embedding_dim=literal_embedding_dim,
            input_dropout=0.1,
        )
        self.shape_kwargs["e"] = literal_embedding_dim
        return kwargs

    def _get_hrt(self, *shapes: Tuple[int, ...]):
        # hotfix for mixed dtypes
        (hc, hl), r, (tc, tl) = super()._get_hrt(*shapes)
        return (hc, hl.real), r, (tc, tl.real)
