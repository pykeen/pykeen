# -*- coding: utf-8 -*-

"""Implementation of the basic utils."""

from poem.constants import EXECUTION_MODE, HPO_MODE, TEST_SET_PATH, TEST_SET_RATIO, KG_ASSUMPTION, OWA, CWA


def is_hpo_mode(config) -> bool:
    """."""
    return config[EXECUTION_MODE] == HPO_MODE

def is_evaluation_requested(config) -> bool:
    return TEST_SET_PATH in config or TEST_SET_RATIO in config

def is_cwa(config):
    """."""
    return config[KG_ASSUMPTION] == config[CWA]