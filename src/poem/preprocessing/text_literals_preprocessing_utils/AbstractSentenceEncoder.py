# -*- coding: utf-8 -*-

"""Abstract class for pre-trained models that encode text passages."""

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    'AbstractSentenceEncoder',
]


class AbstractSentenceEncoder(ABC):
    @abstractmethod
    def encode(self, texts: list) -> np.array:
        """Encode the text passages into vector representations."""
