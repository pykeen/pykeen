# -*- coding: utf-8 -*-

"""Test training mode for ConvE."""

import pykeen.constants as pkc
from tests.constants import BaseTestTrainingMode, set_training_mode_specific_parameters, \
    set_evaluation_specific_parameters


class TestTrainingModeForConvE(BaseTestTrainingMode):
    """Test that ConvE can be trained and evaluated correctly in training mode."""

    config = BaseTestTrainingMode.config
    config = set_training_mode_specific_parameters(config=config)
    config[pkc.KG_EMBEDDING_MODEL_NAME] = pkc.CONV_E_NAME
    config[pkc.EMBEDDING_DIM] = 50
    config[pkc.CONV_E_INPUT_CHANNELS] = 1
    config[pkc.CONV_E_OUTPUT_CHANNELS] = 3
    config[pkc.CONV_E_HEIGHT] = 5
    config[pkc.CONV_E_WIDTH] = 10
    config[pkc.CONV_E_KERNEL_HEIGHT] = 5
    config[pkc.CONV_E_KERNEL_WIDTH] = 3
    config[pkc.CONV_E_INPUT_DROPOUT] = 0.2
    config[pkc.CONV_E_FEATURE_MAP_DROPOUT] = 0.5
    config[pkc.CONV_E_OUTPUT_DROPOUT] = 0.5

    def test_training(self):
        """Test that ConvE is trained correctly in training mode."""
        results = self.execute_pipeline(config=self.config)
        self.check_training_mode_without_evaluation(results=results)

    def test_evaluation(self):
        """Test that ConvE is trained and evaluated correctly in training mode."""
        config = set_evaluation_specific_parameters(config=self.config)
        results = self.execute_pipeline(config=config)
        self.check_training_followed_by_evaluation(results=results)
