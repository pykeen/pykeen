"""Methods around reading and writing of checkpoints."""

import pathlib
from typing import Any, BinaryIO, TypedDict

import torch

from ..models.base import Model

__all__ = [
    "save_model",
    "load_state_torch",
]


class ModelState(TypedDict, total=False):
    """A model state."""

    state_dict: dict[str, Any]


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
    """
    Save a model to a file.

    Example::

        from pykeen.pipeline import pipeline
        from pykeen.checkpoints import save_model, load_state_torch

        result = pipeline(dataset="nations", model="tucker")

        # save model's weights to a file
        save_model(result.model, "/tmp/tucker.pt")
        # load weights again
        state_dict = load_state_torch("/tmp/tucket.pt")
        # update the model
        result.model.load_state_dict(state_dict)
    """
    model_state = get_model_state(model)
    save_state_torch(model_state, file)
