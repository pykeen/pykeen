# -*- coding: utf-8 -*-

"""Optimizers available in POEM.

========  =============================
Name      Reference
========  =============================
adadelta  :class:`torch.optim.Adadelta`
adagrad   :class:`torch.optim.Adagrad`
adam      :class:`torch.optim.Adam`
adamax    :class:`torch.optim.Adamax`
adamw     :class:`torch.optim.AdamW`
sgd       :class:`torch.optim.SGD`
========  =============================

.. note:: This table can be re-generated with ``poem ls optimizers -f rst``
"""

from typing import Mapping, Set, Type, Union

from torch.optim import Adam, SGD
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from .utils import get_cls, normalize_string

__all__ = [
    'optimizers',
    'get_optimizer_cls',
]

_OPTIMIZER_LIST: Set[Type[Optimizer]] = {
    Adam,
    SGD,
    Adagrad,
    Adadelta,
    AdamW,
    Adamax,
}

optimizers: Mapping[str, Type[Optimizer]] = {
    normalize_string(optimizer.__name__): optimizer
    for optimizer in _OPTIMIZER_LIST
}


def get_optimizer_cls(query: Union[None, str, Type[Optimizer]]) -> Type[Optimizer]:
    """Get the optimizer class."""
    return get_cls(
        query,
        base=Optimizer,
        lookup_dict=optimizers,
        default=Adagrad,
    )
