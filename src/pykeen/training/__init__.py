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

from .lcwa import AcceleratedLCWATrainingLoop, LCWATrainingLoop  # noqa: F401
from .slcwa import AcceleratedSLCWATrainingLoop, SLCWATrainingLoop  # noqa: F401
from .training_loop import AcceleratedTrainingLoop, NonFiniteLossError, TrainingLoop  # noqa: F401

__all__ = [
    'TrainingLoop',
    'SLCWATrainingLoop',
    'LCWATrainingLoop',
    'AcceleratedSLCWATrainingLoop',
    'AcceleratedLCWATrainingLoop',
    'NonFiniteLossError',
    'training_loop_resolver',
]

_TRAINING_LOOP_SUFFIX = 'TrainingLoop'
training_loop_resolver = Resolver.from_subclasses(
    base=TrainingLoop,
    default=SLCWATrainingLoop,
    skip={AcceleratedTrainingLoop},
)
