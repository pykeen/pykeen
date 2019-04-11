# -*- coding: utf-8 -*-

"""Test training mode for TransE."""

import pykeen.constants as pkc
from tests.constants import BaseTestTrainingMode


class TestTrainingModeForTransE(BaseTestTrainingMode):
    """Test that TransE can be trained and evaluated correctly in training mode."""
    config = BaseTestTrainingMode.config
    config[pkc.KG_EMBEDDING_MODEL_NAME] = pkc.TRANS_E_NAME
    config[pkc.EMBEDDING_DIM] = 50
    config[pkc.SCORING_FUNCTION_NORM] = 1  # corresponds to L1
    config[pkc.NORM_FOR_NORMALIZATION_OF_ENTITIES] = 2  # corresponds to L2
    config[pkc.MARGIN_LOSS] = 1  # corresponds to L1

    def test_training(self):
        """Test that TransE is trained correctly in training mode."""
        results = self.start_training(config=self.config)
        self.check_basic_results(results=results)
        self.check_that_model_has_not_been_evalauted(results=results)

    def test_evaluation(self):
        """Test that TransE is trained and evaluated correctly in training mode."""
        config = self.set_evaluation_specific_parameters(config=self.config)
        results = self.start_training(config=config)
        self.check_basic_results(results=results)
        self.check_evaluation_results(results=results)
        self.assertIsNotNone(results.results[pkc.FINAL_CONFIGURATION])
