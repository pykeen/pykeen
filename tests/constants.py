# -*- coding: utf-8 -*-

"""Constants for PyKEEN testing."""

import pathlib

__all__ = [
    "HERE",
    "RESOURCES",
    "EPSILON",
]

HERE = pathlib.Path(__file__).resolve().parent
RESOURCES = HERE.joinpath("resources")
EPSILON = 1.0e-07
