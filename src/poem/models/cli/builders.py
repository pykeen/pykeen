# -*- coding: utf-8 -*-

"""Functions for building magical KGE model CLIs."""

import inspect
import json
import logging
import sys
from typing import Optional, Type

import click
from torch import nn

from .options import (
    CLI_OPTIONS, batch_size_option, early_stopping_option, evaluator_option, learning_rate_option, number_epochs_option,
    optimizer_option, testing_option, training_loop_option,
)
from ..base import BaseModule
from ...training import EarlyStopper

__all__ = [
    'build_cli_from_cls',
]

logger = logging.getLogger(__name__)

_OPTIONAL_MAP = {Optional[int]: int, Optional[str]: str}
_SKIP_ANNOTATIONS = {Optional[nn.Embedding], Optional[nn.Parameter], Optional[nn.Module]}


def build_cli_from_cls(model: Type[BaseModule]) -> click.Command:  # noqa: D202
    """Build a :mod:`click` command line interface for a KGE model.

    Allows users to specify all of the (hyper)parameters to the
    model via command line options using :class:`click.Option`.
    """

    def _decorate(command: click.Command) -> click.Command:
        signature = inspect.signature(model.__init__)
        for name, annotation in model.__init__.__annotations__.items():
            if name in {'init', 'return'} or annotation in _SKIP_ANNOTATIONS:
                continue

            if name in CLI_OPTIONS:
                option = CLI_OPTIONS[name]

            elif annotation in {Optional[int], Optional[str]}:
                option = click.option(f'--{name.replace("_", "-")}', type=_OPTIONAL_MAP[annotation])

            else:
                parameter = signature.parameters[name]
                if parameter.default is None:
                    logger.warning(f'Missing handler in {model.__name__} for {name}: '
                                   f'type={annotation} default={parameter.default}')
                    continue

                option = click.option(f'--{name.replace("_", "-")}', type=annotation, default=parameter.default)

            command = option(command)

        return command

    @click.command(help=f'CLI for {model.__name__}', name=model.__name__.lower())
    @optimizer_option
    @training_loop_option
    @number_epochs_option
    @batch_size_option
    @learning_rate_option
    @testing_option
    @evaluator_option
    @early_stopping_option
    @_decorate
    @click.option('--output', type=click.File('w'), default=sys.stdout, help='Where to dump the metric results')
    def main(*, training_loop, optimizer, number_epochs, batch_size, learning_rate, testing, evaluator, early_stopping,
             output, **kwargs):
        """CLI for POEM."""
        click.echo(
            f'Training {model.__name__} with '
            f'{training_loop.__name__[:-len("TrainingLoop")]} using '
            f'{optimizer.__name__} and {evaluator.__name__}'
        )
        model_instance = model(**kwargs)
        training_loop_instance = training_loop(
            model=model_instance,
            optimizer=optimizer(
                params=model_instance.get_grad_params(),
                lr=learning_rate,
            ),
        )
        evaluator_instance = evaluator()

        training_loop_instance.train(
            num_epochs=number_epochs,
            batch_size=batch_size,
            early_stopper=EarlyStopper(
                model=model_instance,
                evaluator=evaluator_instance,
                evaluation_triples_factory=early_stopping,
            ) if early_stopping else None,
        )

        result = evaluator_instance.evaluate(
            model=model_instance,
            mapped_triples=testing and testing.mapped_triples,
        )

        json.dump(result.to_dict(), output, indent=2)
        click.echo('')
        return sys.exit(0)

    return main
