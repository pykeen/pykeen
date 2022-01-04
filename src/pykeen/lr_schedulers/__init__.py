# -*- coding: utf-8 -*-

"""Learning Rate Schedulers available in PyKEEN."""

import dataclasses
from typing import Any, Mapping, Optional, Type, Union

from class_resolver import Resolver
from class_resolver.api import HintOrType
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)
from torch.optim.optimizer import Optimizer

__all__ = [
    "LRScheduler",
    "LRSchedulerWrapper",
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
    "ReduceLROnPlateau",
    "StepLR",
]

LRScheduler = Union[_LRScheduler, ReduceLROnPlateau]

#: The default strategy for optimizing the lr_schedulers' hyper-parameters
lr_schedulers_hpo_defaults: Mapping[Type[_LRScheduler], Mapping[str, Any]] = {
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
        lr_lambda=dict(type="categorical", choices=[lambda epoch: epoch // 30, lambda epoch: 0.95 ** epoch]),
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
    ReduceLROnPlateau: dict(
        factor=dict(type=float, low=1.0e-02, high=1.0 - 1.0e-02, scale="log"),
        mode="min",
        patience=dict(type=int, low=1, high=100, scale="log"),
        threshold=dict(type=float, low=1.0e-06, high=1.0e-02, scale="log"),
    ),
    StepLR: dict(
        gamma=dict(type=float, low=0.1, high=0.9, step=0.1),
        step_size=dict(type=int, low=1, high=50, step=5),
    ),
}

#: A resolver for learning rate schedulers
lr_scheduler_resolver = Resolver(
    base=LRScheduler,
    default=ExponentialLR,
    classes=set(lr_schedulers_hpo_defaults),
)

#: A wrapper around the hidden scheduler base class
@dataclasses.dataclass
class LRSchedulerWrapper:
    """A wrapper around torch's LRScheduler to unify the interface with ReduceLROnPlateau."""

    #: the torch LR scheduler
    base: LRScheduler

    #: the metric name for schedulers which need it
    metric: Optional[str] = None

    @classmethod
    def create(
        cls,
        optimizer: Optimizer,
        base: HintOrType[LRScheduler],
        **kwargs,
    ) -> "LRScheduler":
        """Create a LRScheduler by wrapping a torch LR scheduler."""
        metric = kwargs.pop("metric", None)
        base = lr_scheduler_resolver.make(base=base, optimizer=optimizer, kwargs=kwargs)
        return LRSchedulerWrapper(base=base, metric=metric)

    def recreate_with_optimizer(self, optimizer: Optimizer) -> "LRSchedulerWrapper":
        """Create a new LR scheduler instance."""
        return self.create(
            base=self.base.__class__,
            kwargs=dict(
                optimizer=optimizer,
                metric=self.metric,
                **self.lr_scheduler.get_lr_scheduler_kwargs,
            ),
        )

    @property
    def needs_metrics(self) -> bool:
        """Whether the LRScheduler needs metrics."""
        return self.metric is not None

    def step(self, epoch: int, epoch_loss: float, metric: Optional[float] = None):
        """Step could be called after every batch update."""
        kwargs = dict()
        if self.needs_metrics:
            if self.metric == "loss":
                metric = epoch_loss
            if metric is None:
                raise ValueError(f"LR scheduler {self.base} needs a metric to monitor.")
            kwargs["metrics"] = metric
        return self.base.step(epoch=epoch, **kwargs)

    def get_lr(self):
        """Compute the current learning rate."""
        return self.base.get_lr()

    def state_dict(self) -> Mapping[str, Any]:
        """Return the state of the scheduler as a :class:`dict`."""
        return self.base.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load the scheduler's state."""
        self.base.load_state_dict(state_dict=state_dict)

    def get_lr_scheduler_kwargs(self) -> Mapping[str, Any]:
        """Get the kwargs."""
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("_") and key not in ["base_lrs", "last_epoch"]
        }
