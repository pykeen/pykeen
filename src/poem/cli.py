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

import click
from click_default_group import DefaultGroup
from tabulate import tabulate

from .datasets import data_sets
from .evaluation import evaluators as evaluators_dict, get_metric_list, metrics as metrics_dict
from .hpo.cli import optimize
from .losses import losses as losses_dict
from .models import models as models_dict
from .models.base import BaseModule
from .models.cli import build_cli_from_cls
from .optimizers import optimizers as optimizers_dict
from .regularizers import regularizers as regularizers_dict
from .sampling import negative_samplers as samplers_dict
from .training import training_loops as training_dict
from .utils import get_until_first_blank


@click.group()
def main():
    """POEM."""


tablefmt_option = click.option('-f', '--tablefmt', default='plain', show_default=True)


@main.group(cls=DefaultGroup, default='github-readme', default_if_no_args=True)
def ls():
    """List implementation details."""


@ls.command()
@tablefmt_option
def models(tablefmt: str):
    """List models."""
    lines = list(_get_model_lines(tablefmt=tablefmt))
    headers = ['Name', 'Reference', 'Citation'] if tablefmt in {'rst', 'github'} else ['Name', 'Citation']
    click.echo(
        tabulate(
            lines,
            headers=headers,
            tablefmt=tablefmt,
        ),
    )


def _get_model_lines(tablefmt: str):
    for _, model in sorted(models_dict.items()):
        line = str(model.__doc__.splitlines()[0])
        l, r = line.find('['), line.find(']')
        if tablefmt == 'rst':
            yield model.__name__, f':class:`poem.models.{model.__name__}`', line[l: r + 2]
        elif tablefmt == 'github':
            author, year = line[1 + l: r - 4], line[r - 4: r]
            yield model.__name__, f'`poem.models.{model.__name__}`', f'{author.capitalize()} *et al.*, {year}'
        else:
            author, year = line[1 + l: r - 4], line[r - 4: r]
            yield model.__name__, f'{author.capitalize()}, {year}'


@ls.command()
def parameters():
    """List hyper-parameter usage."""
    click.echo('Names of __init__() parameters in all classes:')

    base_parameters = set(BaseModule.__init__.__annotations__) | {'init'}
    _hyperparemeter_usage = sorted(
        (k, v)
        for k, v in BaseModule._hyperparameter_usage.items()
        if k not in base_parameters
    )
    for i, (name, values) in enumerate(_hyperparemeter_usage, start=1):
        click.echo(f'{i:>2}. {name}')
        for value in sorted(values):
            click.echo(f'    - {value}')


@ls.command()
@tablefmt_option
def datasets(tablefmt: str):
    """List data sets."""
    lines = _get_lines(data_sets, tablefmt, 'datasets')
    click.echo(
        tabulate(
            lines,
            headers=['Name', 'Description'] if tablefmt == 'plain' else ['Name', 'Reference', 'Description'],
            tablefmt=tablefmt,
        ),
    )


@ls.command()
@tablefmt_option
def training(tablefmt: str):
    """List training modes."""
    lines = _get_lines(training_dict, tablefmt, 'training')
    click.echo(
        tabulate(
            lines,
            headers=['Name', 'Description'] if tablefmt == 'plain' else ['Name', 'Reference', 'Description'],
            tablefmt=tablefmt,
        ),
    )


@ls.command()
@tablefmt_option
def samplers(tablefmt: str):
    """List negative samplers."""
    lines = _get_lines(samplers_dict, tablefmt, 'sampling')
    click.echo(
        tabulate(
            lines,
            headers=['Name', 'Description'] if tablefmt == 'plain' else ['Name', 'Reference', 'Description'],
            tablefmt=tablefmt,
        ),
    )


@ls.command()
@tablefmt_option
def evaluators(tablefmt: str):
    """List evaluators."""
    lines = _get_lines(evaluators_dict, tablefmt, 'evaluators')
    click.echo(
        tabulate(
            lines,
            headers=['Name', 'Description'] if tablefmt == 'plain' else ['Name', 'Reference', 'Description'],
            tablefmt=tablefmt,
        ),
    )


@ls.command()
@tablefmt_option
def losses(tablefmt: str):
    """List losses."""
    lines = _get_lines_alternative(tablefmt, losses_dict, 'torch.nn', 'poem.losses')
    click.echo(
        tabulate(
            lines,
            headers=['Name', 'Reference', 'Description'],
            tablefmt=tablefmt,
        ),
    )


@ls.command()
@tablefmt_option
def optimizers(tablefmt: str):
    """List optimizers."""
    lines = _get_lines_alternative(tablefmt, optimizers_dict, 'torch.optim', 'poem.optimizers')
    click.echo(
        tabulate(
            lines,
            headers=['Name', 'Reference', 'Description'],
            tablefmt=tablefmt,
        ),
    )


@ls.command()
@tablefmt_option
def regularizers(tablefmt: str):
    """List regularizers."""
    lines = _get_lines(regularizers_dict, tablefmt, 'regularizers')
    click.echo(
        tabulate(
            lines,
            headers=['Name', 'Reference', 'Description'],
            tablefmt=tablefmt,
        ),
    )


def _get_lines_alternative(tablefmt, d, torch_prefix, poem_prefix):
    for name, submodule in sorted(d.items()):
        if submodule.__module__.startswith('torch'):
            path = f'{torch_prefix}.{submodule.__qualname__}'
        else:  # from poem
            path = f'{poem_prefix}.{submodule.__qualname__}'

        if tablefmt == 'rst':
            yield name, f':class:`{path}`'
        elif tablefmt == 'github':
            doc = submodule.__doc__
            yield name, f'`{path}`', get_until_first_blank(doc)
        else:
            doc = submodule.__doc__
            yield name, path, get_until_first_blank(doc)


@ls.command()
@tablefmt_option
def metrics(tablefmt: str):
    """List metrics."""
    click.echo(
        tabulate(
            _get_metrics_lines(tablefmt),
            headers=['Name', 'Reference'] if tablefmt == 'rst' else ['Metric', 'Description', 'Evaluator', 'Reference'],
            tablefmt=tablefmt,
        ),
    )


def _get_metrics_lines(tablefmt: str):
    if tablefmt == 'rst':
        for name, value in metrics_dict.items():
            yield name, f':class:`poem.evaluation.{value.__name__}`'
    else:
        for field, name, value in get_metric_list():
            if tablefmt == 'github':
                yield (
                    field.name.replace('_', ' ').title(), field.metadata['doc'],
                    name, f'`poem.evaluation.{value.__name__}`',
                )
            else:
                yield field.name, field.metadata['doc'], name, f'poem.evaluation.{value.__name__}',


def _get_lines(d, tablefmt, submodule):
    for name, value in sorted(d.items()):
        if tablefmt == 'rst':
            if isinstance(value, type):
                yield name, f':class:`poem.{submodule}.{value.__name__}`'
            else:
                yield name, f':class:`poem.{submodule}.{name}`'
        elif tablefmt == 'github':
            try:
                ref = value.__name__
                doc = value.__doc__.splitlines()[0]
            except AttributeError:
                ref = name
                doc = value.__class__.__doc__

            yield name, f'`poem.{submodule}.{ref}`', doc
        else:
            yield name, value.__doc__.splitlines()[0]


@ls.command()
@click.pass_context
def github_readme(ctx: click.Context):
    """Generate the GitHub readme's ## Implementation section."""
    click.echo(f'### Models ({len(models_dict)})\n')
    ctx.invoke(models, tablefmt='github')
    click.echo(f'\n### Regularizers ({len(regularizers_dict)})\n')
    ctx.invoke(regularizers, tablefmt='github')
    click.echo(f'\n### Losses ({len(losses_dict)})\n')
    ctx.invoke(losses, tablefmt='github')
    click.echo(f'\n### Data Sets ({len(data_sets)})\n')
    ctx.invoke(datasets, tablefmt='github')
    click.echo(f'\n### Training Modes ({len(training_dict)})\n')
    ctx.invoke(training, tablefmt='github')
    click.echo(f'\n### Optimizers ({len(optimizers_dict)})\n')
    ctx.invoke(optimizers, tablefmt='github')
    click.echo(f'\n### Evaluators ({len(evaluators_dict)})\n')
    ctx.invoke(evaluators, tablefmt='github')
    click.echo(f'\n### Metrics ({len(get_metric_list())})\n')
    ctx.invoke(metrics, tablefmt='github')


@main.group()
@click.pass_context
def train(ctx):
    """Train a KGE model."""


for cls in models_dict.values():
    train.add_command(build_cli_from_cls(cls))

# Add HPO command
main.add_command(optimize)

if __name__ == '__main__':
    main()
