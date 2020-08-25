# -*- coding: utf-8 -*-

"""A validator for experimental settings."""

import inspect
import json
import os
from typing import Iterable, Optional, Set, Type

from torch import nn

from .cli import HERE
from ..datasets import datasets as datasets_dict
from ..losses import _LOSS_SUFFIX, losses as losses_dict
from ..models import models as models_dict
from ..models.base import Model
from ..optimizers import optimizers as optimizers_dict
from ..regularizers import _REGULARIZER_SUFFIX, regularizers as regularizers_dict
from ..sampling import _NEGATIVE_SAMPLER_SUFFIX, negative_samplers as negative_samplers_dict
from ..training import _TRAINING_LOOP_SUFFIX, training_loops as training_loops_dict
from ..utils import normalize_string

_SKIP_NAMES = {
    'loss', 'entity_embeddings', 'init', 'preferred_device', 'random_seed',
    'regularizer', 'relation_embeddings', 'return', 'triples_factory', 'device',
}
_SKIP_ANNOTATIONS = {
    nn.Embedding, Optional[nn.Embedding], Type[nn.Embedding], Optional[Type[nn.Embedding]],
    nn.Module, Optional[nn.Module], Type[nn.Module], Optional[Type[nn.Module]],
    Model, Optional[Model], Type[Model], Optional[Type[Model]],
}


def iterate_config_paths() -> Iterable[str]:
    """Iterate over all configuration paths."""
    for model in os.listdir(HERE):
        if model not in models_dict:
            continue
        model_directory = os.path.join(HERE, model)
        for config in os.listdir(model_directory):
            if config.startswith('hpo'):
                continue
            path = os.path.join(model_directory, config)
            if not os.path.isfile(path) or not path.endswith('.json'):
                continue
            yield model, config, path


def get_configuration_errors(path: str):  # noqa: C901
    """Get a list of errors with a given experimental configuration JSON file."""
    with open(path) as file:
        configuration = json.load(file)

    pipeline = configuration.get('pipeline')
    if pipeline is None:
        raise ValueError('No pipeline')

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
            errors.append(f'No key: {key}')
            return
        if normalize:
            value = normalize_string(value, suffix=suffix)
        if value not in choices:
            errors.append(f'Invalid {key}: {value}. Should be one of {sorted(choices)}')
            return

        if not check_kwargs:
            return

        kwargs_key = f'{key}_kwargs'
        kwargs_value = test_dict.get(kwargs_key)
        if kwargs_value is None:
            errors.append(f'Missing "{kwargs_key}" entry for {value}')
            return

        choice = choices[value]
        signature = inspect.signature(choice.__init__)

        extraneous_kwargs = []
        for name in kwargs_value:
            if name == 'self':
                continue
            if name not in signature.parameters:
                extraneous_kwargs.append(name)
        if extraneous_kwargs:
            _x = '\n'.join(
                f'    {name}'
                for name in extraneous_kwargs
            )
            errors.append(
                f'''Extraneous keys in {kwargs_key} for {choice}:\n{_x}''',
            )

        if allowed_missing_kwargs and required_kwargs:
            raise ValueError('can not specify both allowed and required')

        if allowed_missing_kwargs is None:
            allowed_missing_kwargs = set()

        missing_kwargs = []
        for name, parameter in signature.parameters.items():
            if name == 'self' or parameter.default is inspect._empty or parameter.default is None:
                continue

            annotation = choice.__init__.__annotations__.get(name)

            if name in _SKIP_NAMES or annotation in _SKIP_ANNOTATIONS:
                continue

            if required_kwargs is not None and name not in required_kwargs:
                continue
            if name not in kwargs_value and name not in allowed_missing_kwargs:
                missing_kwargs.append((name, annotation, parameter.default))
        if missing_kwargs:
            _x = '\n'.join(
                f'    {name}: default: {default}, type: {annotation}'
                for name, annotation, default in missing_kwargs
            )
            errors.append(f'Missing {kwargs_key} for {choice}:\n{_x}')

        if extraneous_kwargs or missing_kwargs:
            return

        return value

    _check(
        pipeline, 'model', models_dict,
        normalize=True, check_kwargs=True,
    )
    _check(
        pipeline, 'dataset', datasets_dict,
        normalize=False, check_kwargs=False,
    )
    _check(
        pipeline, 'optimizer', optimizers_dict,
        normalize=True, check_kwargs=True,
        required_kwargs={'lr'},
    )
    _check(
        pipeline, 'loss', losses_dict,
        normalize=True, suffix=_LOSS_SUFFIX, check_kwargs=True,
    )
    _check(
        pipeline, 'regularizer', regularizers_dict,
        normalize=True, suffix=_REGULARIZER_SUFFIX, check_kwargs=True, required=False,
        allowed_missing_kwargs={'dim'},
    )

    training_loop = _check(
        pipeline, 'training_loop', training_loops_dict,
        normalize=True, suffix=_TRAINING_LOOP_SUFFIX, check_kwargs=False,
    )

    if training_loop == 'slcwa':
        _check(
            pipeline, 'negative_sampler', negative_samplers_dict,
            normalize=True, suffix=_NEGATIVE_SAMPLER_SUFFIX, check_kwargs=True,
        )

    return errors
