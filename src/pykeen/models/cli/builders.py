"""Functions for building magical KGE model CLIs."""

import inspect
import json
import logging
import pathlib
import sys
import typing as t
from collections.abc import Mapping
from typing import Any, Optional, Union

import click
from class_resolver import HintOrType
from torch import nn

from . import options
from .options import CLI_OPTIONS
from ..base import Model
from ...nn.message_passing import Decomposition
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import Constrainer, Hint, Initializer, Normalizer

__all__ = [
    "build_cli_from_cls",
]

logger = logging.getLogger(__name__)

_OPTIONAL_MAP = {Optional[int]: int, Optional[str]: str}  # noqa:UP007
_SKIP_ARGS = {
    "return",
    "triples_factory",
    "regularizer",
    # TODO rethink after RGCN update
    "interaction",
    "activation_cls",
    "activation_kwargs",
    "edge_weighting",
    "relation_representations",
    "coefficients",  # from AutoSF
}
_SKIP_ANNOTATIONS = {
    Optional[nn.Embedding],  # noqa:UP007
    Optional[nn.Parameter],  # noqa:UP007
    Optional[nn.Module],  # noqa:UP007
    Optional[Mapping[str, Any]],  # noqa:UP007
    Union[None, str, nn.Module],  # noqa:UP007
    Union[None, str, Decomposition],  # noqa:UP007
}
_SKIP_HINTS = {
    Hint[Initializer],  # type:ignore[misc]
    Hint[Constrainer],  # type:ignore[misc]
    Hint[Normalizer],  # type:ignore[misc]
    Hint[Regularizer],  # type:ignore[misc]
    HintOrType[nn.Module],  # type:ignore[misc]
}


def build_cli_from_cls(model: type[Model]) -> click.Command:  # noqa: D202
    """Build a :mod:`click` command line interface for a KGE model.

    Allows users to specify all of the (hyper)parameters to the model via command line options using
    :class:`click.Option`.

    :param model: the model class

    :returns: a click command for training a model of the given class
    """
    signature = inspect.signature(model.__init__)

    def _decorate_model_kwargs(command: click.decorators.FC) -> click.decorators.FC:
        for name, annotation in model.__init__.__annotations__.items():
            if name in _SKIP_ARGS or annotation in _SKIP_ANNOTATIONS:
                continue

            elif name in CLI_OPTIONS:
                option = CLI_OPTIONS[name]

            elif annotation in {t.Optional[int], t.Optional[str]}:  # noqa:UP007
                option = click.option(f"--{name.replace('_', '-')}", type=_OPTIONAL_MAP[annotation])

            else:
                parameter = signature.parameters[name]
                if annotation in _SKIP_HINTS:
                    logger.debug("Unhandled hint: %s", annotation)
                    continue
                if parameter.default is None:
                    logger.debug(
                        f"Missing handler in {model.__name__} for {name}: "
                        f"type={annotation} default={parameter.default}",
                    )
                    continue

                option = click.option(f"--{name.replace('_', '-')}", type=annotation, default=parameter.default)

            try:
                command = option(command)
            except AttributeError:
                logger.warning(f"Unable to handle parameter in {model.__name__}: {name}")
                continue

        return command

    @click.command(help=f"CLI for {model.__name__}", name=model.__name__.lower())  # type: ignore
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
    @options.inverse_triples_option
    @click.option("--silent", is_flag=True)
    @click.option("--output-directory", type=pathlib.Path, default=None, help="Where to dump the results")
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
        output_directory: pathlib.Path | None,
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
        create_inverse_triples: bool,
        **model_kwargs,
    ):
        """CLI for PyKEEN."""
        click.echo(
            f"Training {model.__name__} with "
            f"{training_loop.__name__.removesuffix('TrainingLoop')} using "
            f"{optimizer.__name__} and {evaluator.__name__}",
        )
        from ...pipeline import pipeline

        result_tracker: str | None
        result_tracker_kwargs: Mapping[str, Any] | None
        if mlflow_tracking_uri:
            result_tracker = "mlflow"
            result_tracker_kwargs = {
                "tracking_uri": mlflow_tracking_uri,
            }
        else:
            result_tracker = None
            result_tracker_kwargs = None

        def _triples_factory(path: str | None) -> TriplesFactory | None:
            if path is None:
                return None
            return TriplesFactory.from_path(path=path, create_inverse_triples=create_inverse_triples)

        training = _triples_factory(training_triples_factory)
        testing = _triples_factory(testing_triples_factory)
        validation = _triples_factory(validation_triples_factory)

        pipeline_result = pipeline(
            device=device,
            model=model,
            model_kwargs=model_kwargs,
            dataset=dataset,
            dataset_kwargs=dict(create_inverse_triples=create_inverse_triples),
            training=training,
            testing=testing or training,
            validation=validation,
            optimizer=optimizer,
            optimizer_kwargs=dict(
                lr=learning_rate,
            ),
            training_loop=training_loop,
            training_loop_kwargs=dict(
                automatic_memory_optimization=automatic_memory_optimization,
            ),
            evaluator=evaluator,
            evaluator_kwargs=dict(),
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
        )
        if output_directory:
            pipeline_result.save_to_directory(
                directory=output_directory,
                # TODO: other parameters?
            )
        elif not silent:
            json.dump(pipeline_result.metric_results.to_dict(), sys.stdout, indent=2)
            click.echo("")

        return sys.exit(0)

    return main
