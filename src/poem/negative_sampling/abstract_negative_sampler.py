# -*- coding: utf-8 -*-

"""Basic structure for a negative sampler."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Mapping

class NegativeSampler(ABC):
    """."""

    @abstractmethod
    def sample(self):
      pass

