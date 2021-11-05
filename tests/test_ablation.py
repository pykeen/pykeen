# -*- coding: utf-8 -*-

"""Tests for ablation studies."""

import tempfile
import unittest

from pykeen.ablation import ablation_pipeline


class TestAblation(unittest.TestCase):
    """Test the ablation pipeline."""

    def test_quate(self):
        """Test using a model with no regularizer argument.

        .. seealso:: https://github.com/pykeen/pykeen/issues/451
        """
        with tempfile.TemporaryDirectory() as directory:
            ablation_pipeline(
                datasets=["nations"],
                models=["quate"],
                directory=directory,
                training_loops=["lcwa", "slcwa"],
                losses=["marginranking"],
                optimizers=["adam"],
                epochs=1,
                n_trials=1,
            )
