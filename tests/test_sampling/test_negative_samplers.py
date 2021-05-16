# -*- coding: utf-8 -*-

"""Test that samplers can be executed."""

from pykeen.sampling import BasicNegativeSampler, BernoulliNegativeSampler, PseudoTypedNegativeSampler
from tests.test_sampling import cases


class BasicNegativeSamplerTest(cases.NegativeSamplerGenericTestCase):
    """Test the basic negative sampler."""

    cls = BasicNegativeSampler

    def test_sample_basic(self):
        """Test if relations and half of heads and tails are not corrupted."""
        negative_batch, batch_filter = self.instance.sample(positive_batch=self.positive_batch)

        # Test that half of the subjects and half of the objects are corrupted
        positive_batch = self.positive_batch.unsqueeze(dim=1)
        num_triples = negative_batch[..., 0].numel()
        half_size = num_triples // 2
        num_subj_corrupted = (positive_batch[..., 0] != negative_batch[..., 0]).sum()
        num_obj_corrupted = (positive_batch[..., 2] != negative_batch[..., 2]).sum()
        assert num_obj_corrupted - 1 <= num_subj_corrupted
        assert num_subj_corrupted - 1 <= num_obj_corrupted
        assert num_subj_corrupted - 1 <= num_triples
        assert half_size - 1 <= num_subj_corrupted


class BernoulliNegativeSamplerTest(cases.NegativeSamplerGenericTestCase):
    """Test the Bernoulli negative sampler."""

    cls = BernoulliNegativeSampler


class PseudoTypedNegativeSamplerTest(cases.NegativeSamplerGenericTestCase):
    """Test the pseudo-type negative sampler."""

    cls = PseudoTypedNegativeSampler
