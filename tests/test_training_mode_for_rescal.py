# -*- coding: utf-8 -*-

"""Test training mode for RESCAL."""

import logging
import os
import tempfile
import unittest

import numpy as np

import pykeen
import pykeen.constants as pkc
from tests.constants import RESOURCES_DIRECTORY, BaseTestTrainingMode

logging.basicConfig(level=logging.INFO)
logging.getLogger('pykeen').setLevel(logging.INFO)




class TestTrainingModeForRESCAL(BaseTestTrainingMode):
    """Test that RESCAL can be trained and evaluated correctly in training mode."""
    config = BaseTestTrainingMode.config
    config[pkc.KG_EMBEDDING_MODEL_NAME] = pkc.RESCAL_NAME
    config[pkc.EMBEDDING_DIM] = 50
    config[pkc.SCORING_FUNCTION_NORM] = 2 # corresponds to L2
    config[pkc.MARGIN_LOSS] = 1 # corresponds to L1

    def test_training(self):
        """Test that RESCAL is trained correctly in training mode."""
        results = self.start_training(config=self.config)
        self.check_basic_results(results=results)
        self.check_that_model_has_not_been_evalauted(results=results)

    def test_evaluation(self):
        """Test that RESCAL is trained and evaluated correctly in training mode. """
        # 10 % of training set will be used as a test set
        config = self.config.copy()
        config[pkc.TEST_SET_RATIO] = 0.1
        config[pkc.FILTER_NEG_TRIPLES] = True

        results = self.start_training(config=config)
        self.check_basic_results(results=results)
        self.check_evaluation_results(results=results)
