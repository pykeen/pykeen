# -*- coding: utf-8 -*-

"""Tests for literal models."""

import unittest

from pykeen.datasets.tnl import NationsLiteralDataset
from pykeen.pipeline import pipeline


class TestLiteralModel(unittest.TestCase):
    """Test that the pipeline can be run on literal datasets."""

    def _help(self, model):
        return pipeline(
            dataset=NationsLiteralDataset,
            model=model,
            training_kwargs=dict(num_epochs=5, use_tqdm=False),
            evaluation_kwargs=dict(use_tqdm=False),
        )

    def test_complex(self):
        """Test running on ComplEx."""
        _ = self._help('ComplExLiteral')

    def test_distmult(self):
        """Test running on DistMult."""
        _ = self._help('DistMultLiteral')
