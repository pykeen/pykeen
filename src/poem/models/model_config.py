# -*- coding: utf-8 -*-

"""Basic dataclass representing the model configuration."""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ModelConfig:
    """Base condiguration for KGE models."""

    #: Configuration dictionary
    config: Dict[str, Any]

    multimodal_data: Dict[str, np.ndarray] = None
