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

from typing import Any, Mapping, Set, Type

from class_resolver import Resolver
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

__all__ = [
    'Optimizer',
    'optimizers_hpo_defaults',
    'optimizer_resolver',
]

_OPTIMIZER_LIST: Set[Type[Optimizer]] = {
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    SGD,
}

#: The default strategy for optimizing the optimizers' hyper-parameters (yo dawg)
optimizers_hpo_defaults: Mapping[Type[Optimizer], Mapping[str, Any]] = {
    Adagrad: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
    Adam: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
    Adamax: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
    AdamW: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
    SGD: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
}

optimizer_resolver = Resolver(_OPTIMIZER_LIST, base=Optimizer, default=Adam)
