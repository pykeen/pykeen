# -*- coding: utf-8 -*-

"""Test pipeline module."""

import unittest

from pykeen.constants import SEED, PREFERRED_DEVICE, HPO_MODE, EXECUTION_MODE, TRAINING_MODE
from pykeen.utilities.pipeline import Pipeline, CPU


CONFIG = {
    SEED:2,
    PREFERRED_DEVICE: CPU
}

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.config = CONFIG
        self.p = Pipeline(config=self.config)

    def test_instantiate_pipeline(self):
        """Test that Pipeline can be instantiated."""
        p = Pipeline(config=CONFIG)
        self.assertIsNotNone(p)

    def test_use_hpo(self):
        """Test whether execution mode is identified correctly."""
        self.p.config[EXECUTION_MODE] = HPO_MODE
        value = self.p._use_hpo(config=self.p.config)
        self.assertTrue(value)

        self.p.config[EXECUTION_MODE] = TRAINING_MODE
        value = self.p._use_hpo(config=self.p.config)
        self.assertFalse(value)