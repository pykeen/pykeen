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

from typing import Set, Type

from class_resolver import Resolver

from .callbacks import TrainingCallback  # noqa: F401
from .lcwa import LCWATrainingLoop  # noqa: F401
from .slcwa import SLCWATrainingLoop  # noqa: F401
from .training_loop import NonFiniteLossError, TrainingLoop  # noqa: F401

__all__ = [
    'TrainingLoop',
    'SLCWATrainingLoop',
    'LCWATrainingLoop',
    'NonFiniteLossError',
    'training_loop_resolver',
    'TrainingCallback',
]

_TRAINING_LOOP_SUFFIX = 'TrainingLoop'
_TRAINING_LOOPS: Set[Type[TrainingLoop]] = {
    LCWATrainingLoop,
    SLCWATrainingLoop,
}
training_loop_resolver = Resolver(
    _TRAINING_LOOPS,
    base=TrainingLoop,  # type: ignore
    default=SLCWATrainingLoop,
    suffix=_TRAINING_LOOP_SUFFIX,
)
