# -*- coding: utf-8 -*-

"""Basic dataclass representing the model configuration."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ModelConfig():
    """."""
    config: Dict
    has_multimodal_data = False
    multimodal_data: Dict[str: np.array]