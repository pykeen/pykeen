# -*- coding: utf-8 -*-

"""A validator for experimental settings."""

import inspect
import pathlib
from typing import Callable, Iterable, Optional, Set, Tuple, Type, Union

import torch
from class_resolver import Hint
from torch import nn

from .cli import HERE
from ..datasets import dataset_resolver
from ..evaluation.ranking_metric_lookup import normalize_flattened_metric_results
from ..losses import loss_resolver
from ..models import Model, model_resolver
from ..optimizers import optimizer_resolver
from ..regularizers import regularizer_resolver
from ..sampling import negative_sampler_resolver
from ..training import training_loop_resolver
from ..utils import CONFIGURATION_FILE_FORMATS, load_configuration, normalize_string

_SKIP_NAMES = {
    "loss",
    "entity_embeddings",
    "init",
    "random_seed",
    "regularizer",
    "relation_embeddings",
    "return",
    "triples_factory",
    "device",
}
_SKIP_ANNOTATIONS = {
    nn.Embedding,
    Optional[nn.Embedding],
    Type[nn.Embedding],
    Optional[Type[nn.Embedding]],
    nn.Module,
    Optional[nn.Module],
    Type[nn.Module],
    Optional[Type[nn.Module]],
    Model,
    Optional[Model],
    Type[Model],
    Optional[Type[Model]],
    Union[str, Callable[[torch.FloatTensor], torch.FloatTensor]],
    Hint[nn.Module],
}
_SKIP_EXTRANEOUS = {
    "predict_with_sigmoid",
}


def iterate_config_paths() -> Iterable[Tuple[str, pathlib.Path, pathlib.Path]]:
    """Iterate over all configuration paths."""
    for model_directory in HERE.iterdir():
        if model_directory.name not in model_resolver.lookup_dict:
            continue
        for config in model_directory.iterdir():
            if config.name.startswith("hpo"):
                continue
            path = model_directory.joinpath(config)
            if not path.is_file() or path.suffix not in CONFIGURATION_FILE_FORMATS:
                continue
            yield model_directory.name, config, path


def _should_skip_because_type(x):
    # don't worry about functions because they can't be specified by JSON.
    # Could make a better mo
    if inspect.isfunction(x):
        return True
    # later could extend for other non-JSON valid types
    return False


def get_configuration_errors(path: Union[str, pathlib.Path]):  # noqa: C901
    """Get a list of errors with a given experimental configuration JSON file."""
    configuration = load_configuration(path)

    pipeline = configuration.get("pipeline")
    if pipeline is None:
        raise ValueError("No pipeline")

    errors = []

    def _check(
        test_dict,
        key,
        choices,
        *,
        required: bool = True,
        normalize: bool = False,
        suffix: Optional[str] = None,
        check_kwargs: bool = False,
        required_kwargs: Optional[Set[str]] = None,
        allowed_missing_kwargs: Optional[Set[str]] = None,
    ):
        value = test_dict.get(key)
        if value is None:
            if not required:
                return
            errors.append(f"No key: {key}")
            return
        if normalize:
            value = normalize_string(value, suffix=suffix)
        if value not in choices:
            errors.append(f"Invalid {key}: {value}. Should be one of {sorted(choices)}")
            return

        if not check_kwargs:
            return

        kwargs_key = f"{key}_kwargs"
        kwargs_value = test_dict.get(kwargs_key)
        if kwargs_value is None:
            errors.append(f'Missing "{kwargs_key}" entry for {value}')
            return

        choice = choices[value]
        signature = inspect.signature(choice.__init__)

        extraneous_kwargs = []
        for name in kwargs_value:
            if name == "self":
                continue
            if name not in signature.parameters and name not in _SKIP_EXTRANEOUS:
                extraneous_kwargs.append(name)
        if extraneous_kwargs:
            _x = "\n".join(f"    {name}" for name in extraneous_kwargs)
            errors.append(
                f"""Extraneous keys in {kwargs_key} for {choice}:\n{_x}""",
            )

        if allowed_missing_kwargs and required_kwargs:
            raise ValueError("can not specify both allowed and required")

        if allowed_missing_kwargs is None:
            allowed_missing_kwargs = set()

        missing_kwargs = []
        for name, parameter in signature.parameters.items():
            if (
                name == "self"
                or parameter.default is inspect._empty  # type:ignore
                or parameter.default is None
            ):
                continue

            annotation = choice.__init__.__annotations__.get(name)

            if name in _SKIP_NAMES or annotation in _SKIP_ANNOTATIONS:
                continue
            if parameter.default and _should_skip_because_type(parameter.default):
                continue

            if required_kwargs is not None and name not in required_kwargs:
                continue
            if name not in kwargs_value and name not in allowed_missing_kwargs:
                missing_kwargs.append((name, annotation, parameter.default))
        if missing_kwargs:
            _x = "\n".join(
                f"    {name}: default: {default}, type: {annotation}" for name, annotation, default in missing_kwargs
            )
            errors.append(f"Missing {kwargs_key} for {choice}:\n{_x}")

        if extraneous_kwargs or missing_kwargs:
            return

        return value

    _check(
        pipeline,
        "model",
        model_resolver.lookup_dict,
        normalize=True,
        check_kwargs=True,
    )
    _check(
        pipeline,
        "dataset",
        dataset_resolver.lookup_dict,
        normalize=False,
        check_kwargs=False,
    )
    _check(
        pipeline,
        "optimizer",
        optimizer_resolver.lookup_dict,
        normalize=True,
        check_kwargs=True,
        required_kwargs={"lr"},
    )
    _check(
        pipeline,
        "loss",
        loss_resolver.lookup_dict,
        normalize=True,
        suffix=loss_resolver.suffix,
        check_kwargs=True,
    )
    _check(
        pipeline,
        "regularizer",
        regularizer_resolver.lookup_dict,
        normalize=True,
        suffix=regularizer_resolver.suffix,
        check_kwargs=True,
        required=False,
        allowed_missing_kwargs={"dim"},
    )

    training_loop = _check(
        pipeline,
        "training_loop",
        training_loop_resolver.lookup_dict,
        normalize=True,
        suffix=training_loop_resolver.suffix,
        check_kwargs=False,
    )

    if training_loop == "slcwa":
        _check(
            pipeline,
            "negative_sampler",
            negative_sampler_resolver.lookup_dict,
            normalize=True,
            suffix=negative_sampler_resolver.suffix,
            check_kwargs=True,
        )

    try:
        normalize_flattened_metric_results(configuration.get("results", {}))
    except ValueError as error:
        errors.append(f"error in parsing results: {error}")

    return errors
