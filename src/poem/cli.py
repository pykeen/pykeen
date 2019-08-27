# -*- coding: utf-8 -*-

"""A command line interface for POEM.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems - the code will get executed twice:

- When you run ``python -m poem`` python will execute``__main__.py`` as a script. That means there won't be any
  ``poem.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``poem.__main__`` in ``sys.modules``.

.. seealso:: http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

from typing import Mapping, Type

import click

from .models.base import BaseModule, RegularizedModel


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


_base_module_subclasses: Mapping[str, Type[BaseModule]] = {
    cls.__name__: cls
    for cls in BaseModule.__subclasses__() + RegularizedModel.__subclasses__()
}


def _model_callback(ctx, param, value: str) -> Type[BaseModule]:
    return _base_module_subclasses[value]


@main.group()
def train():
    """Train a KGE model."""


for cls in _base_module_subclasses.values():
    train.add_command(cls.cli)

if __name__ == '__main__':
    main()
