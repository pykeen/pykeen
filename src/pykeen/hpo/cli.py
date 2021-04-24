# -*- coding: utf-8 -*-

"""A command line interface for hyper-parameter optimization in PyKEEN."""

import sys
from typing import Optional

import click

from .hpo import hpo_pipeline
from .samplers import sampler_resolver
from ..losses import loss_resolver


@click.command()
@click.argument('model')
@click.argument('dataset')
@loss_resolver.get_option('-l', '--loss')
@sampler_resolver.get_option('--sampler', help="Which sampler should be used?")
@click.option('--storage', help="Where to output trials dataframe")
@click.option('--n-trials', type=int, help="Number of trials to run")
@click.option('--timeout', type=int, help="The timeout in seconds")
@click.option('-o', '--output', type=click.Path(file_okay=False, dir_okay=True), help="Where to output results")
def optimize(
    model: str,
    dataset: str,
    loss: str,
    sampler: str,
    storage: Optional[str],
    n_trials: Optional[int],
    timeout: Optional[int],
    output: str,
):
    """Optimize hyper-parameters for a KGE model.

    For example, use pykeen optimize TransE Nations --loss MarginRankingLoss
    """
    if n_trials is None and timeout is None:
        click.secho('Must specify either --n-trials or --timeout', fg='red')
        sys.exit(1)

    hpo_pipeline_result = hpo_pipeline(
        model=model,
        dataset=dataset,
        loss=loss,
        n_trials=n_trials,
        timeout=timeout,
        storage=storage,
        sampler=sampler,
    )
    hpo_pipeline_result.save_to_directory(output)


if __name__ == '__main__':
    optimize()
