"""Utilities for weight averaging."""

import pathlib
from collections.abc import Iterator
from typing import TypeVar

from torch.optim import swa_utils

from pykeen.checkpoints.base import load_state_torch
from pykeen.models.base import Model

__all__ = [
    "exponential_moving_weight_average",
    "stochastic_weight_average",
    "weight_average",
]

M = TypeVar("M", bound=Model)


def weight_average(model: M, checkpoints: Iterator[str | pathlib.Path], decay: float | None = None) -> M:
    """
    Calculate a weight average from the given checkpoints.

    :param model:
        The model, whose weights will be used for temporary storage.
    :param checkpoints:
        An iterator over the paths to stored checkpoints. The order is important for exponential moving average
        weighting.
    :param decay:
        The decay for exponential moving average. If `None`, stochastic weight averaging is used instead.

    .. warning::
        This will overwrite the model's weights in-place.

    .. seealso::
        - https://pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema

    :return:
        A weight-averaged model.
    """
    if decay is None:
        return stochastic_weight_average(model=model, checkpoints=checkpoints)
    return exponential_moving_weight_average(model=model, checkpoints=checkpoints, decay=decay)


def stochastic_weight_average(model: M, checkpoints: Iterator[str | pathlib.Path]) -> M:
    """
    Calculate stochastic weight average from the given checkpoints.

    :param model:
        The model, whose weights will be used for temporary storage.
    :param checkpoints:
        An iterator over the paths to stored checkpoints. The order does not matter.

    .. warning::
        This will overwrite the model's weights in-place.

    .. seealso::
        - https://pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema

    :return:
        a weight-averaged model
    """
    average_model = swa_utils.AveragedModel(model)
    for file in checkpoints:
        state_dict = load_state_torch(file)["state_dict"]
        model.load_state_dict(state_dict)
        average_model.update_parameters(model)
    return average_model.module


def exponential_moving_weight_average(model: M, checkpoints: Iterator[str | pathlib.Path], decay: float = 0.999) -> M:
    """
    Calculate expontential moving weight average from the given checkpoints.

    :param model:
        The model, whose weights will be used for temporary storage.
    :param checkpoints:
        An iterator over the paths to stored checkpoints. The order is important.
    :param decay:
        The decay factor.

    .. warning::
        This will overwrite the model's weights in-place.

    .. seealso::
        - https://pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema

    :return:
        a weight-averaged model
    """
    average_model = swa_utils.AveragedModel(model, multi_avg_fn=swa_utils.get_ema_multi_avg_fn(decay=0.999))
    for file in checkpoints:
        state_dict = load_state_torch(file)["state_dict"]
        model.load_state_dict(state_dict)
        average_model.update_parameters(model)
    return average_model.module
