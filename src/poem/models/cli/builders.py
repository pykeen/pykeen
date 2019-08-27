# -*- coding: utf-8 -*-

"""Functions for building magical KGE model CLIs."""

import json
import sys

import click

from .options import (
    CLI_OPTIONS, batch_size_option, closed_world_option, learning_rate_option, number_epochs_option, optimizer_option,
    testing_option,
)

__all__ = [
    'build_cli_from_cls',
]


def build_cli_from_cls(cls) -> click.Command:  # noqa: D202
    """Build a :mod:`click` command line interface for a KGE model.

    Allows users to specify all of the (hyper)parameters to the
    model via command line options using :class:`click.Option`.

    """

    def decorate(command: click.Command) -> click.Command:
        for name, anno in cls.__init__.__annotations__.items():
            # print(cls, name, anno)
            wrapper = CLI_OPTIONS.get(name)
            if wrapper is not None:
                command = wrapper(command)
        return command

    @click.command(help=f'CLI for {cls.__name__}', name=cls.__name__.lower())
    @optimizer_option
    @closed_world_option
    @number_epochs_option
    @batch_size_option
    @learning_rate_option
    @testing_option
    @decorate
    @click.option('--output', type=click.File('w'), default=sys.stdout)
    def main(*, closed_world, optimizer, number_epochs, batch_size, learning_rate, testing, output, **kwargs):
        """CLI for POEM."""
        from ...evaluation import RankBasedEvaluator
        from ...training import CWATrainingLoop, OWATrainingLoop

        training_loop_cls = CWATrainingLoop if closed_world else OWATrainingLoop
        click.echo(
            f'Training {cls.__name__} with '
            f'{training_loop_cls.__name__[:-len("TrainingLoop")]} using '
            f'{optimizer.__name__}'
        )
        model = cls(**kwargs)
        loop = training_loop_cls(
            model=model,
            optimizer_cls=optimizer,
            optimizer_kwargs={
                'lr': learning_rate,
            },
        )
        loop.train(
            num_epochs=number_epochs,
            batch_size=batch_size,
        )
        evaluator = RankBasedEvaluator(model)

        if testing is None:
            results = evaluator.evaluate_with_training()
        else:
            results = evaluator.evaluate(testing.mapped_triples)

        json.dump(results.to_dict(), output, indent=2)
        click.echo('')
        return sys.exit(0)

    return main
