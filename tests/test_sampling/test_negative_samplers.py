# -*- coding: utf-8 -*-

"""Test that samplers can be executed."""

from pykeen.sampling import BasicNegativeSampler, BernoulliNegativeSampler, PseudoTypedNegativeSampler
from pykeen.sampling.pseudo_type import create_index
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

    def test_corrupt_batch(self):
        """Additional test for corrupt_batch."""
        positive_batch = self.positive_batch.unsqueeze(dim=1)
        negative_batch = self.instance.corrupt_batch(positive_batch=self.positive_batch)
        # same relation
        assert (negative_batch[..., 1] == positive_batch[..., 1]).all()
        # only corruption of a single entity (note: we do not check for exactly 2, since we do not filter).
        assert ((negative_batch == positive_batch).sum(dim=-1) >= 2).all()
        # check that corrupted entities co-occur with the relation in training data
        for entity_pos in (0, 2):
            er_training = {(r, e) for r, e in self.triples_factory.mapped_triples[:, [1, entity_pos]].tolist()}
            er_negative = {(r, e) for r, e in negative_batch.view(-1, 3)[:, [1, entity_pos]].tolist()}
            assert er_negative.issubset(er_training)

    def test_index_structure(self):
        """Test the index structure."""
        data, offsets = create_index(triples_factory=self.triples_factory)
        triples = self.triples_factory.mapped_triples
        for r in range(self.triples_factory.num_relations):
            triples_with_r = triples[triples[:, 1] == r]
            for i, entity_pos in enumerate((0, 2)):
                index_entities = set(data[offsets[2 * r + i]: offsets[2 * r + i + 1]].tolist())
                triple_entities = set(triples_with_r[:, entity_pos].tolist())
                assert index_entities == triple_entities
