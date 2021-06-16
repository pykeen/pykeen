# -*- coding: utf-8 -*-

"""Learning Rate Schedulers available in PyKEEN.

===========================  =============================================================
Name                         Reference
===========================  =============================================================
CosineAnnealingLR            :class:`torch.optim.lr_scheduler.CosineAnnealingLR`
CosineAnnealingWarmRestarts  :class:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`
CyclicLR                     :class:`torch.optim.lr_scheduler.CyclicLR`
ExponentialLR                :class:`torch.optim.lr_scheduler.ExponentialLR`
LambdaLR                     :class:`torch.optim.lr_scheduler.LambdaLR`
MultiplicativeLR             :class:`torch.optim.lr_scheduler.MultiplicativeLR`
MultiStepLR                  :class:`torch.optim.lr_scheduler.MultiStepLR`
OneCycleLR                   :class:`torch.optim.lr_scheduler.OneCycleLR`
StepLR                       :class:`torch.optim.lr_scheduler.StepLR`
===========================  =============================================================

.. note:: This table can be re-generated with ``pykeen ls lr_schedulers -f rst``
"""

from typing import Any, Mapping, Set, Type

from class_resolver import Resolver
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    StepLR,
)

__all__ = [
    '_LRScheduler',
    'lr_schedulers_hpo_defaults',
    'lr_scheduler_resolver',
]

_LR_SCHEDULER_LIST: Set[Type[_LRScheduler]] = {
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    StepLR,
}

#: The default strategy for optimizing the lr_schedulers' hyper-parameters
# TODO: Adjust search space to something reasonable
lr_schedulers_hpo_defaults: Mapping[Type[_LRScheduler], Mapping[str, Any]] = {
    CosineAnnealingLR: dict(
        eta_min=dict(type=float, low=0.001, high=0.1, scale='log'),
        T_max=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
    CosineAnnealingWarmRestarts: dict(
        T_0=dict(type=float, low=0.001, high=0.1, scale='log'),
        T_multi=dict(type=float, low=0.001, high=0.1, scale='log'),
        eta_min=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
    CyclicLR: dict(
        base_lr=dict(type=float, low=0.001, high=0.1, scale='log'),
        max_lr=dict(type=float, low=0.001, high=0.1, scale='log'),
        # TODO: Decide whether all parameters should be available
    ),
    ExponentialLR: dict(
        gamma=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
    LambdaLR: dict(
        lr_lambda=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
    MultiplicativeLR: dict(
        lr_lambda=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
    MultiStepLR: dict(
        gamma=dict(type=float, low=0.001, high=0.1, scale='log'),
        milestones=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
    OneCycleLR: dict(
        max_lr=dict(type=float, low=0.001, high=0.1, scale='log'),
        pct_start=dict(type=float, low=0.001, high=0.1, scale='log'),
        # TODO: Decide whether all parameters should be available
    ),
    StepLR: dict(
        gamma=dict(type=float, low=0.001, high=0.1, scale='log'),
        step_size=dict(type=float, low=0.001, high=0.1, scale='log'),
    ),
}

lr_scheduler_resolver = Resolver(_LR_SCHEDULER_LIST, base=_LRScheduler, default=ExponentialLR)
