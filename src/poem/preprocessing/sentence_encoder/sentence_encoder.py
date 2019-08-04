# -*- coding: utf-8 -*-

"""Abstract class for pre-trained models that encode text passages."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

__all__ = [
    'SentenceEncoder',
]


class SentenceEncoder(ABC):
    """A base sentence encoder."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.array:
        """Encode the text passages into vector representations."""
