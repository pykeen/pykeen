# -*- coding: utf-8 -*-

"""Abstract class for pre-trained models that encode text passages."""

import numpy as np
from abc import ABC, abstractmethod

class AbstractSentenceEncoder(ABC):

    @abstractmethod
    def encode(self, texts:list) -> np.array:
        """Encode the text passages into vector representations."""