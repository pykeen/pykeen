# -*- coding: utf-8 -*-

"""Testing constants for PyKEEN."""

import logging
import os
import tempfile
import unittest

import numpy as np

import pykeen
import pykeen.constants as pkc

logging.basicConfig(level=logging.INFO)
logging.getLogger('pykeen').setLevel(logging.INFO)

__all__ = [
    'RESOURCES_DIRECTORY',
]

HERE = os.path.abspath(os.path.dirname(__file__))
RESOURCES_DIRECTORY = os.path.join(HERE, 'resources')


def set_training_mode_specific_parameters(config):
    """Set hyper-parameter optimization specific specific parameters."""
    config = config.copy()
    config[pkc.EXECUTION_MODE] = pkc.TRAINING_MODE
    return config


def set_hpo_mode_specific_parameters(config):
    """Set hyper-parameter optimization specific specific parameters."""
    config = config.copy()
    config[pkc.EXECUTION_MODE] = pkc.HPO_MODE
    config[pkc.NUM_OF_HPO_ITERS] = 3
    return config


def set_evaluation_specific_parameters(config):
    """Set evaluation specific parameters."""
    # 10 % of training set will be used as a test set
    config = config.copy()
    config[pkc.TEST_SET_RATIO] = 0.1
    config[pkc.FILTER_NEG_TRIPLES] = True
    return config


class BaseTestTrainingMode(unittest.TestCase):
    """Base class for testing the training mode."""
    config = dict()
    config[pkc.TRAINING_SET_PATH] = os.path.join(RESOURCES_DIRECTORY, 'data', 'rdf.nt'),
    config[pkc.SEED] = 0
    config[pkc.LEARNING_RATE] = 0.01
    config[pkc.NUM_EPOCHS] = 10
    config[pkc.BATCH_SIZE] = 64
    config[pkc.PREFERRED_DEVICE] = pkc.CPU

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def execute_pipeline(self, config):
        """Test that ConvE is trained correctly in training mode."""
        results = pykeen.run(
            config=config,
            output_directory=self.dir.name,
        )

        return results

    def check_basic_results(self, results):
        """Test basic functionalities that are always called when a model is trained in training model."""
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.results[pkc.TRAINED_MODEL])
        self.assertIsNotNone(results.results[pkc.LOSSES])
        self.assertIsNotNone(results.results[pkc.ENTITY_TO_EMBEDDING])
        self.assertIsNotNone(results.results[pkc.ENTITY_TO_ID])
        self.assertIsNotNone(results.results[pkc.RELATION_TO_ID])
        self.assertIsNotNone(results.results[pkc.FINAL_CONFIGURATION])

    def check_training_mode_without_evaluation(self, results):
        """Test that model has not been evaluated."""
        self.check_basic_results(results=results)
        self.assertNotIn(pkc.EVAL_SUMMARY, results.results)

    def check_training_followed_by_evaluation(self, results):
        """Test evaluation specific functionalities."""
        self.check_basic_results(results=results)
        self.assertIn(pkc.MEAN_RANK, results.results[pkc.EVAL_SUMMARY])
        self.assertEqual(type(results.results[pkc.EVAL_SUMMARY][pkc.MEAN_RANK]), float)
        self.assertIn(pkc.HITS_AT_K, results.results[pkc.EVAL_SUMMARY])
        self.assertEqual(type(results.results[pkc.EVAL_SUMMARY][pkc.HITS_AT_K][1]), np.float64)
        self.assertEqual(type(results.results[pkc.EVAL_SUMMARY][pkc.HITS_AT_K][3]), np.float64)
        self.assertEqual(type(results.results[pkc.EVAL_SUMMARY][pkc.HITS_AT_K][5]), np.float64)
        self.assertEqual(type(results.results[pkc.EVAL_SUMMARY][pkc.HITS_AT_K][10]), np.float64)
