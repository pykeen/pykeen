# -*- coding: utf-8 -*-

"""Default Hyper-parameter search spaces in PyKEEN that are used package-wide."""

DEFAULT_DROPOUT_HPO_RANGE = dict(type=float, low=0.0, high=0.5, q=0.1)