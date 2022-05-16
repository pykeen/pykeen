# -*- coding: utf-8 -*-

"""Test the integrity of the reproduction experiment scripts."""

import logging
import unittest

from pykeen.experiments.validate import get_configuration_errors, iterate_config_paths

logger = logging.getLogger(__name__)


class TestExperimentIntegrity(unittest.TestCase):
    """Test the integrity of the reproduction experiment scripts."""


def _generate(model, config, path):  # noqa: D202
    """Generate a new test function for the given model/config/path."""

    def _x(test_case: unittest.TestCase):
        errors = get_configuration_errors(path)
        if errors:
            msg = f"Errors found in {model}: {config}:\n" + "\n".join(f"  {error}" for error in errors)
            test_case.fail(msg)

    return _x


def _append():
    """Append all new tests for each configuration."""
    for model, config, path in iterate_config_paths():
        if model == "rgcn":
            logger.warning("Way too much to do for RGCN...")
            continue
        _x = _generate(model, config, path)
        _x.__name__ = f"test_integrity_{model}_{config.stem}"
        setattr(TestExperimentIntegrity, _x.__name__, _x)


_append()
