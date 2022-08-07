# -*- coding: utf-8 -*-

"""Constants for PyKEEN testing."""

import os
import pathlib
import unittest

__all__ = [
    "HERE",
    "RESOURCES",
    "EPSILON",
    "skip_if_windows",
]

HERE = pathlib.Path(__file__).resolve().parent
RESOURCES = HERE.joinpath("resources")
EPSILON = 1.0e-07

skip_if_windows = unittest.skipIf(os.name == "nt", reason="Test doesn't work well on Windows CI")
