# -*- coding: utf-8 -*-

"""Constants and class maps for building magical KGE model CLIs."""

from typing import List, Mapping, Type

from torch.nn import BCELoss, MarginRankingLoss
from torch.optim import Adam, SGD
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.optimizer import Optimizer
from torch.optim.rmsprop import RMSprop

from ...customized_loss_functions import SoftplusLoss
from ...typing import Loss

__all__ = [
    'criteria',
    'criteria_map',
    'optimizers',
    'optimizer_map',
]

criteria: List[Type[Loss]] = [
    MarginRankingLoss,
    BCELoss,
    SoftplusLoss,
]

criteria_map: Mapping[str, Type[Loss]] = {
    criterion.__name__: criterion
    for criterion in criteria
}

optimizers: List[Type[Optimizer]] = [
    SGD,
    Adam,
    Adagrad,
    Adadelta,
    RMSprop,
]
optimizer_map: Mapping[str, Type[Optimizer]] = {
    optimizer.__name__: optimizer
    for optimizer in optimizers
}
