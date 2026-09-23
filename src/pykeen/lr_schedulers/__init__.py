"""Learning Rate Schedulers available in PyKEEN."""

from collections.abc import Mapping
from typing import Any

from class_resolver.contrib.torch import lr_scheduler_resolver
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
]


#: The default strategy for optimizing the lr_schedulers' hyper-parameters,
#: based on :class:`torch.optim.lr_scheduler.LRScheduler`
lr_schedulers_hpo_defaults: Mapping[type[LRScheduler], Mapping[str, Any]] = {
    CosineAnnealingLR: {
        "T_max": {"type": int, "low": 10, "high": 1000, "step": 50},
    },
    CosineAnnealingWarmRestarts: {
        "T_0": {"type": int, "low": 10, "high": 200, "step": 50},
    },
    CyclicLR: {
        "base_lr": {"type": float, "low": 0.001, "high": 0.1, "scale": "log"},
        "max_lr": {"type": float, "low": 0.1, "high": 0.3, "scale": "log"},
    },
    ExponentialLR: {
        "gamma": {"type": float, "low": 0.8, "high": 1.0, "step": 0.025},
    },
    LambdaLR: {
        "lr_lambda": {"type": "categorical", "choices": [lambda epoch: epoch // 30, lambda epoch: 0.95**epoch]},
    },
    MultiplicativeLR: {
        "lr_lambda": {"type": "categorical", "choices": [lambda epoch: 0.85, lambda epoch: 0.9, lambda epoch: 0.95]},
    },
    MultiStepLR: {
        "gamma": {"type": float, "low": 0.1, "high": 0.9, "step": 0.1},
        "milestones": {"type": "categorical", "choices": [75, 130, 190, 240, 370]},
    },
    OneCycleLR: {
        "max_lr": {"type": float, "low": 0.1, "high": 0.3, "scale": "log"},
    },
    StepLR: {
        "gamma": {"type": float, "low": 0.1, "high": 0.9, "step": 0.1},
        "step_size": {"type": int, "low": 1, "high": 50, "step": 5},
    },
}
