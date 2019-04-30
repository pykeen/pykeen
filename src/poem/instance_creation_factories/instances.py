# -*- coding: utf-8 -*-

"""Implementation of basic instance factory which creates just instances based on standard KG triples."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from poem.constants import OWA, CWA


@dataclass
class Instances:
    """."""
    instances: np.ndarray
    entity_to_id: Dict[str, np.ndarray]
    relation_to_id: Dict[str, np.ndarray]
    kg_assumption: str
    multimodal_data: Dict[str, np.ndarray] = None
    has_multimodal_data: bool = False

@dataclass
class OWAInstances(Instances):
    """."""
    kg_assumption:str = OWA

# @dataclass
# class CWAInstances(Instances):
#     """."""
#     labels: np.ndarray
#     kg_assumption:str = CWA