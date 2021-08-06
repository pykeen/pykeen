# -*- coding: utf-8 -*-

"""Test non-parametric baseline models."""

from typing import Any, MutableMapping

import torch
import unittest_templates

from pykeen.datasets import Nations
from pykeen.models import MarginalDistributionBaseline


class MarginalDistributionBaselineTests(unittest_templates.GenericTestCase[MarginalDistributionBaseline]):
    """Tests for MarginalDistributionBaseline."""

    #: The batch size
    batch_size: int = 3

    #: The tested class
    cls = MarginalDistributionBaseline

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        dataset = Nations()
        self.factory = dataset.training
        kwargs["triples_factory"] = self.factory
        return kwargs

    def test_score_t(self):
        """Test score_t."""
        hr_batch = self.factory.mapped_triples[torch.randint(self.factory.num_triples, size=(self.batch_size,))][:, :2]
        scores = self.instance.score_t(hr_batch=hr_batch)
        assert scores.shape == (self.batch_size, self.factory.num_entities)
        # check probability distribution
        assert (0.0 <= scores).all() and (scores <= 1.0).all()
        assert torch.allclose(scores.sum(dim=1), torch.ones(self.batch_size))

    def test_score_h(self):
        """Test score_h."""
        rt_batch = self.factory.mapped_triples[torch.randint(self.factory.num_triples, size=(self.batch_size,))][:, 1:]
        scores = self.instance.score_h(rt_batch=rt_batch)
        assert scores.shape == (self.batch_size, self.factory.num_entities)
        # check probability distribution
        assert (0.0 <= scores).all() and (scores <= 1.0).all()
        assert torch.allclose(scores.sum(dim=1), torch.ones(self.batch_size))


class OnlyRelationMarginalDistributionBaselineTests(MarginalDistributionBaselineTests):
    """Tests for MarginalDistributionBaseline using only the relation margin."""

    kwargs = dict(
        entity_margin=False,
        relation_margin=True,
    )


class OnlyEntityMarginalDistributionBaselineTests(MarginalDistributionBaselineTests):
    """Tests for MarginalDistributionBaseline using only the entity margin."""

    kwargs = dict(
        entity_margin=True,
        relation_margin=False,
    )


class TrivialMarginalDistributionBaselineTests(MarginalDistributionBaselineTests):
    """Tests for MarginalDistributionBaseline not actually using a marginal distribution."""

    kwargs = dict(
        entity_margin=False,
        relation_margin=False,
    )
