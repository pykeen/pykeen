"""Test non-parametric baseline models."""

from collections.abc import Mapping
from typing import Any, ClassVar

import torch

import pykeen.models
from tests import cases


class MarginalDistributionBaselineTests(cases.EvaluationOnlyModelTestCase):
    """Tests for MarginalDistributionBaseline."""

    cls = pykeen.models.MarginalDistributionBaseline

    def _verify(self, scores: torch.FloatTensor):
        # check probability distribution
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()
        assert torch.allclose(scores.sum(dim=1), torch.ones(self.batch_size))


class OnlyRelationMarginalDistributionBaselineTests(MarginalDistributionBaselineTests):
    """Tests for MarginalDistributionBaseline using only the relation margin."""

    kwargs: ClassVar[Mapping[str, Any]] = {
        "entity_margin": False,
        "relation_margin": True,
    }


class OnlyEntityMarginalDistributionBaselineTests(MarginalDistributionBaselineTests):
    """Tests for MarginalDistributionBaseline using only the entity margin."""

    kwargs: ClassVar[Mapping[str, Any]] = {
        "entity_margin": True,
        "relation_margin": False,
    }


class TrivialMarginalDistributionBaselineTests(MarginalDistributionBaselineTests):
    """Tests for MarginalDistributionBaseline not actually using a marginal distribution."""

    kwargs: ClassVar[Mapping[str, Any]] = {
        "entity_margin": False,
        "relation_margin": False,
    }


class SoftInverseTripleBaselineTests(cases.EvaluationOnlyModelTestCase):
    """Tests for soft inverse triples baseline."""

    cls = pykeen.models.SoftInverseTripleBaseline
