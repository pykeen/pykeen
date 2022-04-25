"""Entity Alignment datasets."""

from .openea import OpenEA
from .wk3l import CN3l, WK3l15k, WK3l120k, MTransEDataset

__all__ = [
    "CN3l",
    "MTransEDataset",
    "OpenEA",
    "WK3l15k",
    "WK3l120k",
]
