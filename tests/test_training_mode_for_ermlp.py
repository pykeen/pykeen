# -*- coding: utf-8 -*-

"""Test training mode for ERMLP."""

import pykeen.constants as pkc
from tests.constants import BaseTestTrainingMode, set_training_mode_specific_parameters, \
    set_evaluation_specific_parameters


class TestTrainingModeForERMLP(BaseTestTrainingMode):
    """Test that UM can be trained and evaluated correctly in training mode."""
    config = BaseTestTrainingMode.config
    config = set_training_mode_specific_parameters(config=config)
    config[pkc.KG_EMBEDDING_MODEL_NAME] = pkc.ERMLP_NAME
    config[pkc.EMBEDDING_DIM] = 50
    config[pkc.MARGIN_LOSS] = 1

    def test_training(self):
        """Test that ERMLP is trained correctly in training mode."""
        results = self.execute_pipeline(config=self.config)
        self.check_basic_results(results=results)
        self.check_that_model_has_not_been_evalauted(results=results)

    def test_evaluation(self):
        """Test that ERMLP is trained and evaluated correctly in training mode."""
        config = set_evaluation_specific_parameters(config=self.config)
        results = self.execute_pipeline(config=config)
        self.check_basic_results(results=results)
        self.check_evaluation_results(results=results)
        self.assertIsNotNone(results.results[pkc.FINAL_CONFIGURATION])
