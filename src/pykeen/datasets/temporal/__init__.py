# -*- coding: utf-8 -*-

"""Datasets for tKGs."""

from .base import TemporalPathDataset
from .gdelt import GDELTm10
from .icews import ICEWS14, ICEWS5to15, ICEWS11to14, SmallSample
from .kg_data import WN18RR, Kinships

__all__ = [
    "TemporalPathDataset",
    "GDELTm10",
    "ICEWS14",
    "ICEWS5to15",
    "ICEWS11to14",
    "WN18RR",
    "Kinships",
    "SmallSample",
]
