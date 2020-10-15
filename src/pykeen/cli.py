# -*- coding: utf-8 -*-

"""A command line interface for PyKEEN.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems - the code will get executed twice:

- When you run ``python -m pykeen`` python will execute``__main__.py`` as a script. That means there won't be any
  ``pykeen.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``pykeen.__main__`` in ``sys.modules``.

.. seealso:: http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

import inspect
import os
import sys
from itertools import chain

import click
from click_default_group import DefaultGroup
from tabulate import tabulate

from .datasets import datasets as datasets_dict
from .evaluation import evaluators as evaluators_dict, get_metric_list, metrics as metrics_dict
from .experiments.cli import experiments
from .hpo.cli import optimize
from .hpo.samplers import samplers as hpo_samplers_dict
from .losses import losses as losses_dict
from .models import models as models_dict
from .models.base import EntityEmbeddingModel, EntityRelationEmbeddingModel, Model
from .models.cli import build_cli_from_cls
from .optimizers import optimizers as optimizers_dict
from .regularizers import regularizers as regularizers_dict
from .sampling import negative_samplers as negative_samplers_dict
from .stoppers import stoppers as stoppers_dict
from .trackers import trackers as trackers_dict
from .training import training_loops as training_dict
from .triples.utils import EXTENSION_IMPORTERS, PREFIX_IMPORTERS
from .utils import get_until_first_blank

HERE = os.path.abspath(os.path.dirname(__file__))


@click.group()
def main():
    """PyKEEN."""


tablefmt_option = click.option('-f', '--tablefmt', default='plain', show_default=True)


@main.group(cls=DefaultGroup, default='github-readme', default_if_no_args=True)
def ls():
    """List implementation details."""


@ls.command()
@tablefmt_option
def models(tablefmt: str):
    """List models."""
    click.echo(_help_models(tablefmt))


def _help_models(tablefmt):
    lines = list(_get_model_lines(tablefmt=tablefmt))
    headers = ['Name', 'Reference', 'Citation'] if tablefmt in {'rst', 'github'} else ['Name', 'Citation']
    return tabulate(
        lines,
        headers=headers,
        tablefmt=tablefmt,
    )


def _get_model_lines(tablefmt: str):
    for _, model in sorted(models_dict.items()):
        line = str(model.__doc__.splitlines()[0])
        l, r = line.find('['), line.find(']')
        if tablefmt == 'rst':
            yield model.__name__, f':class:`pykeen.models.{model.__name__}`', line[l: r + 2]
        elif tablefmt == 'github':
            author, year = line[1 + l: r - 4], line[r - 4: r]
            yield model.__name__, f'`pykeen.models.{model.__name__}`', f'{author.capitalize()} *et al.*, {year}'
        else:
            author, year = line[1 + l: r - 4], line[r - 4: r]
            yield model.__name__, f'{author.capitalize()}, {year}'


@ls.command()
def parameters():
    """List hyper-parameter usage."""
    click.echo('Names of __init__() parameters in all classes:')

    base_parameters = set(chain(
        Model.__init__.__annotations__,
        EntityEmbeddingModel.__init__.__annotations__,
        EntityRelationEmbeddingModel.__init__.__annotations__,
    ))
    _hyperparameter_usage = sorted(
        (k, v)
        for k, v in Model._hyperparameter_usage.items()
        if k not in base_parameters
    )
    for i, (name, values) in enumerate(_hyperparameter_usage, start=1):
        click.echo(f'{i:>2}. {name}')
        for value in sorted(values):
            click.echo(f'    - {value}')


@ls.command()
def importers():
    """List triple importers."""
    for prefix, f in sorted(PREFIX_IMPORTERS.items()):
        click.secho(f'prefix: {prefix} from {inspect.getmodule(f).__name__}')
    for suffix, f in sorted(EXTENSION_IMPORTERS.items()):
        click.secho(f'suffix: {suffix} from {inspect.getmodule(f).__name__}')


@ls.command()
@tablefmt_option
def datasets(tablefmt: str):
    """List datasets."""
    click.echo(_help_datasets(tablefmt))


def _help_datasets(tablefmt):
    lines = _get_lines(datasets_dict, tablefmt, 'datasets')
    return tabulate(
        lines,
        headers=['Name', 'Description'] if tablefmt == 'plain' else ['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def training_loops(tablefmt: str):
    """List training approaches."""
    click.echo(_help_training(tablefmt))


def _help_training(tablefmt):
    lines = _get_lines(training_dict, tablefmt, 'training')
    return tabulate(
        lines,
        headers=['Name', 'Description'] if tablefmt == 'plain' else ['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def negative_samplers(tablefmt: str):
    """List negative samplers."""
    click.echo(_help_negative_samplers(tablefmt))


def _help_negative_samplers(tablefmt):
    lines = _get_lines(negative_samplers_dict, tablefmt, 'sampling')
    return tabulate(
        lines,
        headers=['Name', 'Description'] if tablefmt == 'plain' else ['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def stoppers(tablefmt: str):
    """List stoppers."""
    click.echo(_help_stoppers(tablefmt))


def _help_stoppers(tablefmt):
    lines = _get_lines(stoppers_dict, tablefmt, 'stoppers')
    return tabulate(
        lines,
        headers=['Name', 'Description'] if tablefmt == 'plain' else ['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def evaluators(tablefmt: str):
    """List evaluators."""
    click.echo(_help_evaluators(tablefmt))


def _help_evaluators(tablefmt):
    lines = sorted(_get_lines(evaluators_dict, tablefmt, 'evaluation'))
    return tabulate(
        lines,
        headers=['Name', 'Description'] if tablefmt == 'plain' else ['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def losses(tablefmt: str):
    """List losses."""
    click.echo(_help_losses(tablefmt))


def _help_losses(tablefmt):
    lines = _get_lines_alternative(tablefmt, losses_dict, 'torch.nn', 'pykeen.losses')
    return tabulate(
        lines,
        headers=['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def optimizers(tablefmt: str):
    """List optimizers."""
    click.echo(_help_optimizers(tablefmt))


def _help_optimizers(tablefmt):
    lines = _get_lines_alternative(tablefmt, optimizers_dict, 'torch.optim', 'pykeen.optimizers')
    return tabulate(
        lines,
        headers=['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def regularizers(tablefmt: str):
    """List regularizers."""
    click.echo(_help_regularizers(tablefmt))


def _help_regularizers(tablefmt):
    lines = _get_lines(regularizers_dict, tablefmt, 'regularizers')
    return tabulate(
        lines,
        headers=['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


def _get_lines_alternative(tablefmt, d, torch_prefix, pykeen_prefix):
    for name, submodule in sorted(d.items()):
        if any(
            submodule.__module__.startswith(_prefix)
            for _prefix in ('torch', 'optuna')
        ):
            path = f'{torch_prefix}.{submodule.__qualname__}'
        else:  # from pykeen
            path = f'{pykeen_prefix}.{submodule.__qualname__}'

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
    click.echo(_help_metrics(tablefmt))


def _help_metrics(tablefmt):
    return tabulate(
        sorted(_get_metrics_lines(tablefmt)),
        headers=['Name', 'Reference'] if tablefmt == 'rst' else ['Metric', 'Description', 'Evaluator', 'Reference'],
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def trackers(tablefmt: str):
    """List trackers."""
    click.echo(_help_trackers(tablefmt))


def _help_trackers(tablefmt):
    lines = _get_lines(trackers_dict, tablefmt, 'trackers')
    return tabulate(
        lines,
        headers=['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def hpo_samplers(tablefmt: str):
    """List HPO samplers."""
    click.echo(_help_hpo_samplers(tablefmt))


def _help_hpo_samplers(tablefmt):
    lines = _get_lines_alternative(tablefmt, hpo_samplers_dict, 'optuna.samplers', 'pykeen.hpo.samplers')
    return tabulate(
        lines,
        headers=['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


def _get_metrics_lines(tablefmt: str):
    if tablefmt == 'rst':
        for name, value in metrics_dict.items():
            yield name, f':class:`pykeen.evaluation.{value.__name__}`'
    else:
        for field, name, value in get_metric_list():
            if tablefmt == 'github':
                yield (
                    field.name.replace('_', ' ').title(), field.metadata['doc'],
                    name, f'`pykeen.evaluation.{value.__name__}`',
                )
            else:
                yield field.name, field.metadata['doc'], name, f'pykeen.evaluation.{value.__name__}'


def _get_lines(d, tablefmt, submodule):
    for name, value in sorted(d.items()):
        if tablefmt == 'rst':
            if isinstance(value, type):
                yield name, f':class:`pykeen.{submodule}.{value.__name__}`'
            else:
                yield name, f':class:`pykeen.{submodule}.{name}`'
        elif tablefmt == 'github':
            try:
                ref = value.__name__
                doc = value.__doc__.splitlines()[0]
            except AttributeError:
                ref = name
                doc = value.__class__.__doc__

            yield name, f'`pykeen.{submodule}.{ref}`', doc
        else:
            yield name, value.__doc__.splitlines()[0]


@main.command()
@click.option('--check', is_flag=True)
def readme(check: bool):
    """Generate the GitHub readme's ## Implementation section."""
    readme_path = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'README.md'))
    new_readme = get_readme()

    if check:
        with open(readme_path) as file:
            old_readme = file.read()
        if new_readme.strip() != old_readme.strip():
            click.secho(
                'Readme has not been updated properly! Make sure all changes are made in the template first,'
                ' and see the following diff:',
                fg='red',
            )
            import difflib
            for x in difflib.context_diff(new_readme.splitlines(), old_readme.splitlines()):
                click.echo(x)

            sys.exit(-1)

    with open(readme_path, 'w') as file:
        print(new_readme, file=file)


def get_readme() -> str:
    """Get the readme."""
    from jinja2 import FileSystemLoader, Environment
    loader = FileSystemLoader(os.path.join(HERE, 'templates'))
    environment = Environment(
        autoescape=True,
        loader=loader,
        trim_blocks=False,
    )
    readme_template = environment.get_template('README.md')
    tablefmt = 'github'
    return readme_template.render(
        models=_help_models(tablefmt),
        n_models=len(models_dict),
        regularizers=_help_regularizers(tablefmt),
        n_regularizers=len(regularizers_dict),
        losses=_help_losses(tablefmt),
        n_losses=len(losses_dict),
        datasets=_help_datasets(tablefmt),
        n_datasets=len(datasets_dict),
        training_loops=_help_training(tablefmt),
        n_training_loops=len(training_dict),
        negative_samplers=_help_negative_samplers(tablefmt),
        n_negative_samplers=len(negative_samplers_dict),
        optimizers=_help_optimizers(tablefmt),
        n_optimizers=len(optimizers_dict),
        stoppers=_help_stoppers(tablefmt),
        n_stoppers=len(stoppers_dict),
        evaluators=_help_evaluators(tablefmt),
        n_evaluators=len(evaluators_dict),
        metrics=_help_metrics(tablefmt),
        n_metrics=len(get_metric_list()),
        trackers=_help_trackers(tablefmt),
        n_trackers=len(trackers_dict),
        hpo_samplers=_help_hpo_samplers(tablefmt),
        n_hpo_samplers=len(hpo_samplers_dict),
    )


@main.group()
@click.pass_context
def train(ctx):
    """Train a KGE model."""


for cls in models_dict.values():
    train.add_command(build_cli_from_cls(cls))

# Add HPO command
main.add_command(optimize)
main.add_command(experiments)

if __name__ == '__main__':
    main()
