# -*- coding: utf-8 -*-

"""Training loops for KGE models using multi-modal information.

======  ==========================================
Name    Reference
======  ==========================================
lcwa    :class:`pykeen.training.LCWATrainingLoop`
slcwa   :class:`pykeen.training.SLCWATrainingLoop`
======  ==========================================

.. note:: This table can be re-generated with ``pykeen ls trainers -f rst``
"""

from class_resolver import Resolver
from .callbacks import TrainingCallback  # noqa: F401
from .lcwa import AcceleratedLCWATrainingLoop, LCWATrainingLoop, LCWATrainingLoop  # noqa: F401
from .slcwa import AcceleratedSLCWATrainingLoop, SLCWATrainingLoop, SLCWATrainingLoop  # noqa: F401
from .training_loop import (  # noqa: F401
    AcceleratedTrainingLoop, NonFiniteLossError, NonFiniteLossError, TrainingLoop,
    TrainingLoop,
)

__all__ = [
    'TrainingLoop',
    'SLCWATrainingLoop',
    'LCWATrainingLoop',
    'AcceleratedSLCWATrainingLoop',
    'AcceleratedLCWATrainingLoop',
    'NonFiniteLossError',
    'training_loop_resolver',
    'TrainingCallback',
]

_TRAINING_LOOP_SUFFIX = 'TrainingLoop'
training_loop_resolver = Resolver.from_subclasses(
    base=TrainingLoop,
    default=SLCWATrainingLoop,
    skip={AcceleratedTrainingLoop},
)
