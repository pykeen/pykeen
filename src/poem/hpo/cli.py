# -*- coding: utf-8 -*-

"""A command line interface for hyper-parameter optimization in POEM."""

import json
import sys
from typing import Optional, TextIO

import click

from .hpo import make_study
from .samplers import samplers


@click.command()
@click.argument('model')
@click.option('-d', '--data-set', help="What data set to use", required=True)
@click.option('-l', '--loss')
@click.option('--sampler', help="Which sampler should be used?", type=click.Choice(list(samplers)), default='tpe')
@click.option('--storage', help="Where to output trials dataframe")
@click.option('--n-trials', type=int, help="Number of trials to run")
@click.option('--timeout', type=int, help="Number of trials to run")
@click.option('-o', '--output', type=click.File('w'), help="Where to output trials dataframe")
def optimize(
    model: str,
    data_set: str,
    loss: Optional[str],
    sampler: Optional[str],
    storage: Optional[str],
    n_trials: Optional[int],
    timeout: Optional[int],
    output: Optional[TextIO],
):
    """Optimize hyper-parameters for a KGE model.

    For example, use python -m poem.hpo TransE MarginRankingLoss -d Nations
    """
    if n_trials is None and timeout is None:
        click.secho('Must specify either --n-trials or --timeout', fg='red')
        sys.exit(1)

    study = make_study(
        model=model,
        loss=loss,
        data_set=data_set,
        n_trials=n_trials,
        timeout=timeout,
        storage=storage,
        sampler=sampler,
    )
    click.echo(json.dumps(
        {
            'value': study.best_value,
            'params': study.best_params,
        },
        indent=2,
    ))

    if output is not None:
        df = study.trials_dataframe()
        df.to_csv(output, sep='\t', index=False)


if __name__ == '__main__':
    optimize()
