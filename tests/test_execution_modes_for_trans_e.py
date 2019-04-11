# -*- coding: utf-8 -*-

"""Test training and HPO mode for TransE."""

import pykeen.constants as pkc
from tests.constants import BaseTestTrainingMode, set_training_mode_specific_parameters, \
    set_hpo_mode_specific_parameters, set_evaluation_specific_parameters


class TestTrainingModeForTransE(BaseTestTrainingMode):
    """Test that TransE can be trained and evaluated correctly in training mode."""
    config = BaseTestTrainingMode.config
    config = set_training_mode_specific_parameters(config=config)
    config[pkc.KG_EMBEDDING_MODEL_NAME] = pkc.TRANS_E_NAME
    config[pkc.EMBEDDING_DIM] = 50
    config[pkc.SCORING_FUNCTION_NORM] = 1  # corresponds to L1
    config[pkc.NORM_FOR_NORMALIZATION_OF_ENTITIES] = 2  # corresponds to L2
    config[pkc.MARGIN_LOSS] = 1  # corresponds to L1

    def test_training(self):
        """Test that TransE is trained correctly in training mode."""
        results = self.execute_pipeline(config=self.config)
        self.check_training_mode_without_evaluation(results=results)

    def test_evaluation(self):
        """Test that TransE is trained and evaluated correctly in training mode."""
        config = set_evaluation_specific_parameters(config=self.config)
        results = self.execute_pipeline(config=config)
        self.check_training_followed_by_evaluation(results=results)
        self.assertIsNotNone(results.results[pkc.FINAL_CONFIGURATION])


class TestHPOModeForTransE(BaseTestTrainingMode):
    """Test that TransE can be trained and evaluated correctly in HPO mode."""
    config = BaseTestTrainingMode.config
    config = set_training_mode_specific_parameters(config=config)
    config[pkc.KG_EMBEDDING_MODEL_NAME] = pkc.TRANS_E_NAME
    config[pkc.EMBEDDING_DIM] = [10, 20, 50]
    config[pkc.SCORING_FUNCTION_NORM] = [1, 2]
    config[pkc.NORM_FOR_NORMALIZATION_OF_ENTITIES] = [2]
    config[pkc.MARGIN_LOSS] = [1, 1.5, 2]

    def test_hpo_mode(self):
        """Test whether HPO mode works correctly for TransE."""
        config = set_hpo_mode_specific_parameters(config=self.config)
        results = self.execute_pipeline(config=config)
        self.check_training_followed_by_evaluation(results=results)
