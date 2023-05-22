# -*- coding: utf-8 -*-

"""Datasets for tKGs."""

from .base import TemporalPathDataset
from .icews import ICEWS14, SmallSample

__all__ = [
    "TemporalPathDataset",
    "ICEWS14",
    "SmallSample",
]
