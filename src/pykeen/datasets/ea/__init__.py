"""Entity Alignment datasets."""

from .base import EADataset
from .openea import OpenEA
from .wk3l import CN3l, MTransEDataset, WK3l15k, WK3l120k

__all__ = [
    # base
    "EADataset",
    # concrete
    "CN3l",
    "MTransEDataset",
    "OpenEA",
    "WK3l15k",
    "WK3l120k",
]
