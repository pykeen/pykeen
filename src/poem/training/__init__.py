# -*- coding: utf-8 -*-

"""Training loops for KGE models using multi-modal information.

======  ==========================
Name    Reference
======  ==========================
cwa     :class:`poem.training.cwa`
owa     :class:`poem.training.owa`
======  ==========================

.. note:: This table can be re-generated with ``poem ls training -f rst``
"""

from typing import Type, Union

from .cwa import CWATrainingLoop  # noqa: F401
from .early_stopping import EarlyStopper  # noqa: F401
from .owa import OWATrainingLoop  # noqa: F401
from .training_loop import TrainingLoop  # noqa: F401
from ..utils import get_cls

__all__ = [
    'TrainingLoop',
    'OWATrainingLoop',
    'CWATrainingLoop',
    'EarlyStopper',
    'training_loops',
    'get_training_loop_cls',
]

#: A mapping of training loops' names to their implementations
training_loops = {
    'owa': OWATrainingLoop,
    'cwa': CWATrainingLoop,
}


def get_training_loop_cls(query: Union[None, str, Type[TrainingLoop]]) -> Type[TrainingLoop]:
    """Get the training loop class."""
    return get_cls(
        query,
        base=TrainingLoop,
        lookup_dict=training_loops,
        default=OWATrainingLoop,
    )
