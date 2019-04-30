# -*- coding: utf-8 -*-

"""Basic dataclass representing the model configuration."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ModelConfig:
    """."""
    config: Dict
    multimodal_data: Dict[str,np.ndarray] = None
    has_multimodal_data = False
