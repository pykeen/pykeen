# -*- coding: utf-8 -*-

"""Training loops for KGE models using multi-modal information.

======  =======================================
Name    Reference
======  =======================================
lcwa    :class:`poem.training.LCWATrainingLoop`
owa     :class:`poem.training.OWATrainingLoop`
======  =======================================

.. note:: This table can be re-generated with ``poem ls training -f rst``
"""

from typing import Mapping, Set, Type, Union

from .early_stopping import EarlyStopper  # noqa: F401
from .lcwa import LCWATrainingLoop  # noqa: F401
from .owa import OWATrainingLoop  # noqa: F401
from .training_loop import TrainingLoop  # noqa: F401
from ..utils import get_cls, normalize_string

__all__ = [
    'TrainingLoop',
    'OWATrainingLoop',
    'LCWATrainingLoop',
    'EarlyStopper',
    'training_loops',
    'get_training_loop_cls',
]

_TRAINING_LOOP_SUFFIX = 'TrainingLoop'
_TRAINING_LOOPS: Set[Type[TrainingLoop]] = {
    LCWATrainingLoop,
    OWATrainingLoop,
}

#: A mapping of training loops' names to their implementations
training_loops: Mapping[str, Type[TrainingLoop]] = {
    normalize_string(cls.__name__, suffix=_TRAINING_LOOP_SUFFIX): cls
    for cls in _TRAINING_LOOPS
}


def get_training_loop_cls(query: Union[None, str, Type[TrainingLoop]]) -> Type[TrainingLoop]:
    """Get the training loop class."""
    return get_cls(
        query,
        base=TrainingLoop,
        lookup_dict=training_loops,
        default=OWATrainingLoop,
        suffix=_TRAINING_LOOP_SUFFIX,
    )
