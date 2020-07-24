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

from typing import Mapping, Set, Type, Union

from .lcwa import LCWATrainingLoop  # noqa: F401
from .slcwa import SLCWATrainingLoop  # noqa: F401
from .training_loop import NonFiniteLossError, TrainingLoop  # noqa: F401
from ..utils import get_cls, normalize_string

__all__ = [
    'TrainingLoop',
    'SLCWATrainingLoop',
    'LCWATrainingLoop',
    'NonFiniteLossError',
    'training_loops',
    'get_training_loop_cls',
]

_TRAINING_LOOP_SUFFIX = 'TrainingLoop'
_TRAINING_LOOPS: Set[Type[TrainingLoop]] = {
    LCWATrainingLoop,
    SLCWATrainingLoop,
}

#: A mapping of training loops' names to their implementations
training_loops: Mapping[str, Type[TrainingLoop]] = {
    normalize_string(cls.__name__, suffix=_TRAINING_LOOP_SUFFIX): cls
    for cls in _TRAINING_LOOPS
}


def get_training_loop_cls(query: Union[None, str, Type[TrainingLoop]]) -> Type[TrainingLoop]:
    """Look up a training loop class by name (case/punctuation insensitive) in :data:`pykeen.training.training_loops`.

    :param query: The name of the training loop (case insensitive, punctuation insensitive).
    :return: The training loop class
    """
    return get_cls(
        query,
        base=TrainingLoop,
        lookup_dict=training_loops,
        default=SLCWATrainingLoop,
        suffix=_TRAINING_LOOP_SUFFIX,
    )
