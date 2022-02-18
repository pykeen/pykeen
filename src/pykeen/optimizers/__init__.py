# -*- coding: utf-8 -*-

"""Optimizers available in PyKEEN."""

from typing import Any, Mapping, Type

from class_resolver.contrib.torch import optimizer_resolver
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

__all__ = [
    "Optimizer",
    "optimizers_hpo_defaults",
    "optimizer_resolver",
]

#: The default strategy for optimizing the optimizers' hyper-parameters (yo dawg)
optimizers_hpo_defaults: Mapping[Type[Optimizer], Mapping[str, Any]] = {
    Adagrad: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale="log"),
    ),
    Adam: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale="log"),
    ),
    Adamax: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale="log"),
    ),
    AdamW: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale="log"),
    ),
    SGD: dict(
        lr=dict(type=float, low=0.001, high=0.1, scale="log"),
    ),
}
