# -*- coding: utf-8 -*-

"""Type hints for POEM."""

from typing import Optional

from torch import nn

__all__ = [
    'OptionalLoss',
]

OptionalLoss = Optional[nn.modules.loss._Loss]
