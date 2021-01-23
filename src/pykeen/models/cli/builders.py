# -*- coding: utf-8 -*-

"""Functions for building magical KGE model CLIs."""

import inspect
import json
import logging
import sys
from typing import Optional, Type

import click
from torch import nn

from . import options
from .options import CLI_OPTIONS
from ..base import Model

__all__ = [
    'build_cli_from_cls',
]

logger = logging.getLogger(__name__)

_OPTIONAL_MAP = {Optional[int]: int, Optional[str]: str}
_SKIP_ARGS = {
    'return',
    'triples_factory',
    'preferred_device',
    'regularizer',
    # TODO rethink after RGCN update
    'activation_cls',
    'activation_kwargs',
    'edge_weighting'
}
_SKIP_ANNOTATIONS = {
    Optional[nn.Embedding],
    Optional[nn.Parameter],
    Optional[nn.Module],
}


def build_cli_from_cls(model: Type[Model]) -> click.Command:  # noqa: D202
    """Build a :mod:`click` command line interface for a KGE model.

    Allows users to specify all of the (hyper)parameters to the
    model via command line options using :class:`click.Option`.
    """
    signature = inspect.signature(model.__init__)

    def _decorate_model_kwargs(command: click.Command) -> click.Command:
        for name, annotation in model.__init__.__annotations__.items():
            if name in _SKIP_ARGS or annotation in _SKIP_ANNOTATIONS:
                continue

            elif name in CLI_OPTIONS:
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

            try:
                command = option(command)
            except AttributeError:
                logger.warning(f'Unable to handle parameter in {model.__name__}: {name}')
                continue

        return command

    @click.command(help=f'CLI for {model.__name__}', name=model.__name__.lower())
    @options.device_option
    @options.dataset_option
    @options.training_option
    @options.testing_option
    @options.valiadation_option
    @options.optimizer_option
    @options.training_loop_option
    @options.automatic_memory_optimization_option
    @options.number_epochs_option
    @options.batch_size_option
    @options.learning_rate_option
    @options.evaluator_option
    @options.stopper_option
    @options.mlflow_uri_option
    @options.title_option
    @options.num_workers_option
    @options.random_seed_option
    @_decorate_model_kwargs
    @click.option('--silent', is_flag=True)
    @click.option('--output', type=click.File('w'), default=sys.stdout, help='Where to dump the metric results')
    def main(
        *,
        device,
        training_loop,
        optimizer,
        number_epochs,
        batch_size,
        learning_rate,
        evaluator,
        stopper,
        output,
        mlflow_tracking_uri,
        title,
        dataset,
        automatic_memory_optimization,
        training_triples_factory,
        testing_triples_factory,
        validation_triples_factory,
        num_workers,
        random_seed,
        silent: bool,
        **model_kwargs,
    ):
        """CLI for PyKEEN."""
        click.echo(
            f'Training {model.__name__} with '
            f'{training_loop.__name__[:-len("TrainingLoop")]} using '
            f'{optimizer.__name__} and {evaluator.__name__}',
        )
        from ...pipeline import pipeline

        if mlflow_tracking_uri:
            result_tracker = 'mlflow'
            result_tracker_kwargs = {
                'tracking_uri': mlflow_tracking_uri,
            }
        else:
            result_tracker = None
            result_tracker_kwargs = None

        pipeline_result = pipeline(
            device=device,
            model=model,
            model_kwargs=model_kwargs,
            dataset=dataset,
            training=training_triples_factory,
            testing=testing_triples_factory or training_triples_factory,
            validation=validation_triples_factory,
            optimizer=optimizer,
            optimizer_kwargs=dict(
                lr=learning_rate,
            ),
            training_loop=training_loop,
            evaluator=evaluator,
            training_kwargs=dict(
                num_epochs=number_epochs,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            stopper=stopper,
            result_tracker=result_tracker,
            result_tracker_kwargs=result_tracker_kwargs,
            metadata=dict(
                title=title,
            ),
            random_seed=random_seed,
            automatic_memory_optimization=automatic_memory_optimization,
        )

        if not silent:
            json.dump(pipeline_result.metric_results.to_dict(), output, indent=2)
            click.echo('')
        return sys.exit(0)

    return main
