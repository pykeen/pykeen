# -*- coding: utf-8 -*-

"""Implementation of the basic utils."""

from poem.constants import CWA, EXECUTION_MODE, HPO_MODE, TEST_SET_PATH, TEST_SET_RATIO, KG_ASSUMPTION


def is_hpo_mode(config) -> bool:
    """Check if execution mode is HPO."""
    return config[EXECUTION_MODE] == HPO_MODE


def is_evaluation_requested(config) -> bool:
    """Check if evaluation is necessary."""
    return TEST_SET_PATH in config or TEST_SET_RATIO in config


def is_cwa(config) -> bool:
    """Check if closed_world_assumption is KG_assumption."""
    return config[KG_ASSUMPTION] == config[CWA]
