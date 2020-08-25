# -*- coding: utf-8 -*-

"""A command line interface for hyper-parameter optimization in PyKEEN."""

import sys
from typing import Optional

import click

from .hpo import hpo_pipeline
from .samplers import samplers


@click.command()
@click.argument('model')
@click.argument('dataset')
@click.option('-l', '--loss')
@click.option(
    '--sampler', help="Which sampler should be used?", type=click.Choice(list(samplers)), default='tpe',
    show_default=True,
)
@click.option('--storage', help="Where to output trials dataframe")
@click.option('--n-trials', type=int, help="Number of trials to run")
@click.option('--timeout', type=int, help="Number of trials to run")
@click.option('-o', '--output', type=click.Path(file_okay=False, dir_okay=True), help="Where to output results")
def optimize(
    model: str,
    dataset: str,
    loss: Optional[str],
    sampler: Optional[str],
    storage: Optional[str],
    n_trials: Optional[int],
    timeout: Optional[int],
    output: str,
):
    """Optimize hyper-parameters for a KGE model.

    For example, use python -m pykeen.hpo TransE MarginRankingLoss -d Nations
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
