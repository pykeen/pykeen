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
from .lcwa import LCWATrainingLoop, _LCWATrainingLoop  # noqa: F401
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

training_loop_resolver = Resolver.from_subclasses(
    base=TrainingLoop,  # type: ignore
    default=SLCWATrainingLoop,
    skip={
        _LCWATrainingLoop,
    },
)
