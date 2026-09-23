"""Utilities for image lookup, caching, and encoding."""

from .cache import WikidataImageCache
from .representation import VisionDataset, VisualRepresentation, WikidataVisualRepresentation

__all__ = [
    "VisionDataset",
    "VisualRepresentation",
    "WikidataVisualRepresentation",
    "WikidataImageCache",
]
