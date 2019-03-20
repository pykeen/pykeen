# -*- coding: utf-8 -*-

"""Test pipeline module."""

import json
import os
import unittest

from pykeen.constants import (
    EXECUTION_MODE,
    HPO_MODE,
    PREFERRED_DEVICE,
    SEED,
    TEST_SET_PATH,
    TEST_SET_RATIO,
    TRAINING_MODE,
    CPU)
from pykeen.utilities.pipeline import Pipeline

CONFIG = {
    SEED: 2,
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

    def test_is_evaluation_required(self):
        """Test whether evaluation option is identified correctly."""
        self.p.config[TEST_SET_PATH] = '/test/path'
        value = self.p.is_evaluation_required
        self.assertTrue(value)

        del self.p.config[TEST_SET_PATH]

        self.p.config[TEST_SET_RATIO] = 0.1
        value = self.p.is_evaluation_required
        self.assertTrue(value)

        del self.p.config[TEST_SET_RATIO]

        value = self.p.is_evaluation_required
        self.assertFalse(value)

    def test_run(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        config_path = os.path.join(
            os.path.abspath(os.path.join(dir_path, os.pardir)),
            'test_resources',
            'configuration_training_without_eval.json'
        )

        with open(config_path) as json_data:
            config = json.load(json_data)

        config['training_set_path'] = os.path.join(
            os.path.abspath(os.path.join(dir_path, os.pardir)),
            'test_resources',
            'example_training.tsv'
        )

        config['test_set_path'] = os.path.join(
            os.path.abspath(os.path.join(dir_path, os.pardir)),
            'test_resources',
            'example_test.tsv'
        )

        self.p.config = config
        results = self.p.run()
        self.assertIsNotNone(results)


        config_path = os.path.join(
            os.path.abspath(os.path.join(dir_path, os.pardir)),
            'test_resources',
            'configuration_hpo.json'
        )

        with open(config_path) as json_data:
            config = json.load(json_data)

        config['training_set_path'] = os.path.join(
            os.path.abspath(os.path.join(dir_path, os.pardir)),
            'test_resources',
            'example_training.tsv'
        )

        config['test_set_path'] = os.path.join(
            os.path.abspath(os.path.join(dir_path, os.pardir)),
            'test_resources',
            'example_test.tsv'
        )

        self.p.config = config
        results = self.p.run()
        self.assertIsNotNone(results)