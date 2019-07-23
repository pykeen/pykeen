# -*- coding: utf-8 -*-

"""A command line interface for POEM."""

import click

from .models import BaseModule


@click.group()
def main():
    """POEM."""


@main.command()
def parameters():
    """List hyper-parameter usage."""
    click.echo('Names of init variables in all classes:')
    for i, (name, values) in enumerate(sorted(BaseModule._hyperparameter_usage.items()), start=1):
        click.echo(f'{i:>2}. {name}')
        for value in sorted(values):
            click.echo(f'    - {value}')


if __name__ == '__main__':
    main()
