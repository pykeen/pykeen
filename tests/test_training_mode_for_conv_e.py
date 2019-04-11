# -*- coding: utf-8 -*-

"""Test training mode for ConvE."""

import logging

import pykeen.constants as pkc
from tests.constants import BaseTestTrainingMode

logging.basicConfig(level=logging.INFO)
logging.getLogger('pykeen').setLevel(logging.INFO)

CONFIG = dict(

    margin_loss=1,
    learning_rate=0.01,
    num_epochs=20,
    batch_size=64,
    preferred_device='cpu'
)


class TestTrainingModeForConvE(BaseTestTrainingMode):
    """Test that ConvE can be trained and evaluated correctly in training mode."""

    config = BaseTestTrainingMode.config
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
        results = self.start_training(config=self.config)
        self.check_basic_results(results=results)
        self.check_that_model_has_not_been_evalauted(results=results)

    def test_evaluation(self):
        """Test that ConvE is trained and evaluated correctly in training mode. """
        # 10 % of training set will be used as a test set
        config = self.config.copy()
        config[pkc.TEST_SET_RATIO] = 0.1
        config[pkc.FILTER_NEG_TRIPLES] = True

        results = self.start_training(config=config)
        self.check_basic_results(results=results)
        self.check_evaluation_results(results=results)
