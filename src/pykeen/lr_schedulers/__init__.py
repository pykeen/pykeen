# -*- coding: utf-8 -*-

"""Learning Rate Schedulers available in PyKEEN."""

from typing import Any, Mapping, Type

from class_resolver import ClassResolver
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LRScheduler,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    StepLR,
)

__all__ = [
    "LRScheduler",
    "lr_schedulers_hpo_defaults",
    "lr_scheduler_resolver",
    # Imported from PyTorch
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "ExponentialLR",
    "LambdaLR",
    "MultiplicativeLR",
    "MultiStepLR",
    "OneCycleLR",
    "StepLR",
]

# fixme: bring this upstream to class_resolver.contrib?
lr_scheduler_resolver = ClassResolver.from_subclasses(LRScheduler, default=ExponentialLR, suffix="LR")

#: The default strategy for optimizing the lr_schedulers' hyper-parameters
lr_schedulers_hpo_defaults: Mapping[Type[LRScheduler], Mapping[str, Any]] = {
    CosineAnnealingLR: dict(
        T_max=dict(type=int, low=10, high=1000, step=50),
    ),
    CosineAnnealingWarmRestarts: dict(
        T_0=dict(type=int, low=10, high=200, step=50),
    ),
    CyclicLR: dict(
        base_lr=dict(type=float, low=0.001, high=0.1, scale="log"),
        max_lr=dict(type=float, low=0.1, high=0.3, scale="log"),
    ),
    ExponentialLR: dict(
        gamma=dict(type=float, low=0.8, high=1.0, step=0.025),
    ),
    LambdaLR: dict(
        lr_lambda=dict(type="categorical", choices=[lambda epoch: epoch // 30, lambda epoch: 0.95**epoch]),
    ),
    MultiplicativeLR: dict(
        lr_lambda=dict(type="categorical", choices=[lambda epoch: 0.85, lambda epoch: 0.9, lambda epoch: 0.95]),
    ),
    MultiStepLR: dict(
        gamma=dict(type=float, low=0.1, high=0.9, step=0.1),
        milestones=dict(type="categorical", choices=[75, 130, 190, 240, 370]),
    ),
    OneCycleLR: dict(
        max_lr=dict(type=float, low=0.1, high=0.3, scale="log"),
    ),
    StepLR: dict(
        gamma=dict(type=float, low=0.1, high=0.9, step=0.1),
        step_size=dict(type=int, low=1, high=50, step=5),
    ),
}
