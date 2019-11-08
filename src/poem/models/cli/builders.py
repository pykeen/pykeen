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
    CLI_OPTIONS, batch_size_option, early_stopping_option, evaluator_option, learning_rate_option, mlflow_uri_option,
    number_epochs_option, optimizer_option, testing_option, title_option, training_loop_option, training_option,
)
from ..base import BaseModule

__all__ = [
    'build_cli_from_cls',
]

logger = logging.getLogger(__name__)

_OPTIONAL_MAP = {Optional[int]: int, Optional[str]: str}
_SKIP_ARGS = {'init', 'return', 'triples_factory'}
_SKIP_ANNOTATIONS = {Optional[nn.Embedding], Optional[nn.Parameter], Optional[nn.Module]}


def build_cli_from_cls(model: Type[BaseModule]) -> click.Command:  # noqa: D202
    """Build a :mod:`click` command line interface for a KGE model.

    Allows users to specify all of the (hyper)parameters to the
    model via command line options using :class:`click.Option`.
    """

    def _decorate(command: click.Command) -> click.Command:
        signature = inspect.signature(model.__init__)
        for name, annotation in model.__init__.__annotations__.items():
            if name in _SKIP_ARGS or annotation in _SKIP_ANNOTATIONS:
                continue

            if name in CLI_OPTIONS:
                option = CLI_OPTIONS[name]

            elif annotation in {Optional[int], Optional[str]}:
                option = click.option(f'--{name.replace("_", "-")}', type=_OPTIONAL_MAP[annotation])

            else:
                parameter = signature.parameters[name]
                if parameter.default is None:
                    logger.warning(
                        f'Missing handler in {model.__name__} for {name}: '
                        f'type={annotation} default={parameter.default}',
                    )
                    continue

                option = click.option(f'--{name.replace("_", "-")}', type=annotation, default=parameter.default)

            command = option(command)

        return command

    @click.command(help=f'CLI for {model.__name__}', name=model.__name__.lower())
    @training_option
    @testing_option
    @optimizer_option
    @training_loop_option
    @number_epochs_option
    @batch_size_option
    @learning_rate_option
    @evaluator_option
    @early_stopping_option
    @mlflow_uri_option
    @title_option
    @_decorate
    @click.option('--output', type=click.File('w'), default=sys.stdout, help='Where to dump the metric results')
    def main(
        *, training_loop, optimizer, number_epochs, batch_size, learning_rate, evaluator, early_stopping,
        output, mlflow_tracking_uri, title, training_triples_factory, testing_triples_factory, **model_kwargs,
    ):
        """CLI for POEM."""
        click.echo(
            f'Training {model.__name__} with '
            f'{training_loop.__name__[:-len("TrainingLoop")]} using '
            f'{optimizer.__name__} and {evaluator.__name__}',
        )
        from ...pipeline import pipeline

        pipeline_result = pipeline(
            model=model,
            model_kwargs=model_kwargs,
            training_triples_factory=training_triples_factory,
            testing_triples_factory=testing_triples_factory or training_triples_factory,
            optimizer=optimizer,
            optimizer_kwargs=dict(
                lr=learning_rate,
            ),
            training_loop=training_loop,
            evaluator=evaluator,
            training_kwargs=dict(
                num_epochs=number_epochs,
                batch_size=batch_size,
            ),
            early_stopping=early_stopping,
            mlflow_tracking_uri=mlflow_tracking_uri,
            metadata=dict(
                title=title,
            ),
        )

        json.dump(pipeline_result.metric_results.to_dict(), output, indent=2)
        click.echo('')
        return sys.exit(0)

    return main
