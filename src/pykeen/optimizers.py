# -*- coding: utf-8 -*-

"""Optimizers available in PyKEEN.

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

.. note:: This table can be re-generated with ``pykeen ls optimizers -f rst``
"""

from typing import Any, Mapping, Set, Type, Union

from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

from .utils import get_cls, normalize_string

__all__ = [
    'Optimizer',
    'optimizers',
    'optimizers_hpo_defaults',
    'get_optimizer_cls',
]

_OPTIMIZER_LIST: Set[Type[Optimizer]] = {
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    SGD,
}

#: A mapping of optimizers' names to their implementations
optimizers: Mapping[str, Type[Optimizer]] = {
    normalize_string(optimizer.__name__): optimizer
    for optimizer in _OPTIMIZER_LIST
}

#: The default strategy for optimizing the optimizers' hyper-parameters (yo dawg)
optimizers_hpo_defaults: Mapping[Type[Optimizer], Mapping[str, Any]] = {
    Adadelta: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
        weight_decay=dict(type=float, low=0., high=1.0, q=0.1),
    ),
    Adagrad: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
        lr_decay=dict(type=float, low=0.001, high=0.1, scale='log'),
        weight_decay=dict(type=float, low=0., high=1.0, q=0.1),
    ),
    Adam: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
        weight_decay=dict(type=float, low=0., high=1.0, q=0.1),
    ),
    Adamax: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
        weight_decay=dict(type=float, low=0., high=1.0, q=0.1),
    ),
    AdamW: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
        weight_decay=dict(type=float, low=0., high=1.0, q=0.1),
    ),
    SGD: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
        weight_decay=dict(type=float, low=0., high=1.0, q=0.1),
    ),
}


def get_optimizer_cls(query: Union[None, str, Type[Optimizer]]) -> Type[Optimizer]:
    """Get the optimizer class."""
    return get_cls(
        query,
        base=Optimizer,
        lookup_dict=optimizers,
        default=Adagrad,
    )
