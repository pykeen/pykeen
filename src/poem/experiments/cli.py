# -*- coding: utf-8 -*-

"""Run landmark experiments."""

import json

import click

from poem.experiments import run_bordes2013_transe_fb15k, run_example_experiment
from poem.pipeline import PipelineResult


@click.group()
def main():
    """Run landmark experiments."""


@main.command()
def example():
    """Run an example experiment."""
    result: PipelineResult = run_example_experiment()
    click.echo(json.dumps(result.metric_results.to_dict(), indent=2))


@main.command()
def bordes2013():
    """Run the bordes2013 experiment."""
    result: PipelineResult = run_bordes2013_transe_fb15k()
    click.echo(json.dumps(result.metric_results.to_dict(), indent=2))


if __name__ == '__main__':
    main()
