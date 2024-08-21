"""Methods around reading and writing of checkpoints."""

import pathlib
from typing import Any, BinaryIO, TypedDict

import torch

from ..models.base import Model


class ModelState(TypedDict, total=False):
    """A model state."""

    state_dict: dict[str, Any]
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    configuration: dict[str, Any]


def get_model_state(model: Model) -> ModelState:
    """Get a serializable representation of the model's state."""
    # TODO: without label to id mapping a model might be pretty use-less
    # TODO: it would be nice to get a configuration to re-construct the model
    return {"state_dict": model.state_dict()}


def save_state_torch(state: ModelState, file: pathlib.Path | str | BinaryIO) -> None:
    """Write a state using PyTorch."""
    torch.save(state, file)


def load_state_torch(file: pathlib.Path | str | BinaryIO) -> ModelState:
    """Read a state using PyTorch."""
    state = torch.load(file)
    return state


def save_model(model: Model, file: pathlib.Path | str | BinaryIO) -> None:
    """Save a model."""
    model_state = get_model_state(model)
    save_state_torch(model_state, file)
