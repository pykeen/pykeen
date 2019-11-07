# -*- coding: utf-8 -*-

"""Unittest for for global utilities."""

import unittest

import numpy
import torch

from poem.utils import get_until_first_blank, l2_regularization


class L2RegularizationTest(unittest.TestCase):
    """Test L2 regularization."""

    def test_one_tensor(self):
        """Test if output is correct for a single tensor."""
        t = torch.ones(1, 2, 3, 4)
        reg = l2_regularization(t)
        self.assertAlmostEqual(float(reg), float(numpy.prod(t.shape)))

    def test_many_tensors(self):
        """Test if output is correct for var-args."""
        ts = []
        exp_reg = 0.
        for i, shape in enumerate([
            (1, 2, 3),
            (2, 3, 4),
            (3, 4, 5),
        ]):
            t = torch.ones(*shape) * (i + 1)
            ts.append(t)
            exp_reg += numpy.prod(t.shape) * (i + 1) ** 2
        reg = l2_regularization(*ts)
        self.assertAlmostEqual(float(reg), exp_reg)


class TestGetUntilFirstBlank(unittest.TestCase):
    """Test get_until_first_blank()."""

    def test_get_until_first_blank_trivial(self):
        """Test the trivial string."""
        s = ''
        r = get_until_first_blank(s)
        self.assertEqual('', r)

    def test_regular(self):
        """Test a regulat case."""
        s = """Broken
        line.

        Now I continue.
        """
        r = get_until_first_blank(s)
        self.assertEqual("Broken line.", r)
