# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..constants import CWA, OWA

__all__ = [
    'Instances',
    'OWAInstances',
    'CWAInstances',
    'MultimodalInstances',
    'MultimodalOWAInstances',
    'MultimodalCWAInstances',
]


@dataclass
class Instances:
    """Triples and mappings to their indices."""

    instances: np.ndarray
    entity_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]


@dataclass
class OWAInstances(Instances):
    """Triples and mappings to their indices for OWA."""

    kg_assumption: str = OWA


@dataclass
class CWAInstances(Instances):
    """Triples and mappings to their indices for CWA."""

    labels: np.ndarray
    kg_assumption: str = CWA


@dataclass
class MultimodalInstances(Instances):
    """Triples and mappings to their indices as well as multimodal data."""

    multimodal_data: Dict[str, np.ndarray]
    data_relation_to_id: Dict[str, int]


@dataclass
class MultimodalOWAInstances(OWAInstances, MultimodalInstances):
    """Triples and mappings to their indices as well as multimodal data for OWA."""


@dataclass
class MultimodalCWAInstances(CWAInstances, MultimodalInstances):
    """Triples and mappings to their indices as well as multimodal data for CWA."""
