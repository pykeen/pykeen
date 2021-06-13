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
from pathlib import Path
from typing import Optional

import click
from click_default_group import DefaultGroup
from tabulate import tabulate

from .datasets import dataset_resolver
from .evaluation import evaluator_resolver, get_metric_list, metric_resolver
from .experiments.cli import experiments
from .hpo.cli import optimize
from .hpo.samplers import sampler_resolver
from .losses import loss_resolver
from .models import model_resolver
from .models.cli import build_cli_from_cls
from .optimizers import optimizer_resolver
from .regularizers import regularizer_resolver
from .sampling import negative_sampler_resolver
from .stoppers import stopper_resolver
from .trackers import tracker_resolver
from .training import training_loop_resolver
from .triples.utils import EXTENSION_IMPORTERS, PREFIX_IMPORTERS
from .utils import get_until_first_blank
from .version import env_table

HERE = Path(__file__).resolve().parent


@click.group()
def main():
    """PyKEEN."""


@main.command()
@click.option('-f', '--tablefmt', default='github', show_default=True)
def version(tablefmt):
    """Print version information for debugging."""
    click.echo(env_table(tablefmt))


tablefmt_option = click.option('-f', '--tablefmt', default='plain', show_default=True)


@main.group(cls=DefaultGroup, default='github-readme', default_if_no_args=True)
def ls():
    """List implementation details."""


@ls.command()
@tablefmt_option
def models(tablefmt: str):
    """List models."""
    click.echo(_help_models(tablefmt))


def _help_models(tablefmt: str, link_fmt: Optional[str] = None):
    lines = list(_get_model_lines(tablefmt=tablefmt, link_fmt=link_fmt))
    headers = ['Name', 'Reference', 'Citation'] if tablefmt in {'rst', 'github'} else ['Name', 'Citation']
    return tabulate(
        lines,
        headers=headers,
        tablefmt=tablefmt,
    )


def _get_model_lines(tablefmt: str, link_fmt: Optional[str] = None):
    for _, model in sorted(model_resolver.lookup_dict.items()):
        reference = f'pykeen.models.{model.__name__}'
        docdata = getattr(model, '__docdata__', None)
        if docdata is not None:
            if link_fmt:
                reference = f'[`{reference}`]({link_fmt.format(reference)})'
            else:
                reference = f'`{reference}`'
            citation = docdata['citation']
            citation_str = f"[{citation['author']} *et al.*, {citation['year']}]({citation['link']})"
            yield model.__name__, reference, citation_str
        else:
            line = str(model.__doc__.splitlines()[0])
            l, r = line.find('['), line.find(']')
            if tablefmt == 'rst':
                yield model.__name__, f':class:`{reference}`', line[l: r + 2]
            elif tablefmt == 'github':
                author, year = line[1 + l: r - 4], line[r - 4: r]
                if link_fmt:
                    reference = f'[`{reference}`]({link_fmt.format(reference)})'
                else:
                    reference = f'`{reference}`'
                yield model.__name__, reference, f'{author.capitalize()} *et al.*, {year}'
            else:
                author, year = line[1 + l: r - 4], line[r - 4: r]
                yield model.__name__, f'{author.capitalize()}, {year}'


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


def _help_datasets(tablefmt: str, link_fmt: Optional[str] = None):
    lines = _get_dataset_lines(tablefmt=tablefmt, link_fmt=link_fmt)
    return tabulate(
        lines,
        headers=['Name', 'Documentation', 'Citation', 'Entities', 'Relations', 'Triples'],
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def training_loops(tablefmt: str):
    """List training approaches."""
    click.echo(_help_training(tablefmt))


def _help_training(tablefmt: str, link_fmt: Optional[str] = None):
    lines = _get_lines(training_loop_resolver.lookup_dict, tablefmt, 'training', link_fmt=link_fmt)
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


def _help_negative_samplers(tablefmt: str, link_fmt: Optional[str] = None):
    lines = _get_lines(negative_sampler_resolver.lookup_dict, tablefmt, 'sampling', link_fmt=link_fmt)
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


def _help_stoppers(tablefmt: str, link_fmt: Optional[str] = None):
    lines = _get_lines(stopper_resolver.lookup_dict, tablefmt, 'stoppers', link_fmt=link_fmt)
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


def _help_evaluators(tablefmt, link_fmt: Optional[str] = None):
    lines = sorted(_get_lines(evaluator_resolver.lookup_dict, tablefmt, 'evaluation', link_fmt=link_fmt))
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


def _help_losses(tablefmt: str, link_fmt: Optional[str] = None):
    lines = _get_lines_alternative(tablefmt, loss_resolver.lookup_dict, 'torch.nn', 'pykeen.losses', link_fmt)
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


def _help_optimizers(tablefmt: str, link_fmt: Optional[str] = None):
    lines = _get_lines_alternative(
        tablefmt, optimizer_resolver.lookup_dict, 'torch.optim', 'pykeen.optimizers',
        link_fmt=link_fmt,
    )
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


def _help_regularizers(tablefmt, link_fmt: Optional[str] = None):
    lines = _get_lines(regularizer_resolver.lookup_dict, tablefmt, 'regularizers', link_fmt=link_fmt)
    return tabulate(
        lines,
        headers=['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


def _get_lines_alternative(tablefmt, d, torch_prefix, pykeen_prefix, link_fmt: Optional[str] = None):
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
            if link_fmt:
                reference = f'[`{path}`]({link_fmt.format(path)})'
            else:
                reference = f'`{path}`'

            yield name, reference, get_until_first_blank(doc)
        else:
            doc = submodule.__doc__
            yield name, path, get_until_first_blank(doc)


@ls.command()
@tablefmt_option
def metrics(tablefmt: str):
    """List metrics."""
    click.echo(_help_metrics(tablefmt))


def _help_metrics(tablefmt, link_fmt=None):
    return tabulate(
        sorted(_get_metrics_lines(tablefmt, link_fmt=link_fmt)),
        headers=(
            ['Name', 'Reference'] if tablefmt == 'rst'
            else ['Name', 'Description'] if tablefmt == 'github'
            else ['Metric', 'Description', 'Reference']
        ),
        tablefmt=tablefmt,
    )


@ls.command()
@tablefmt_option
def trackers(tablefmt: str):
    """List trackers."""
    click.echo(_help_trackers(tablefmt))


def _help_trackers(tablefmt: str, link_fmt: Optional[str] = None):
    lines = _get_lines(tracker_resolver.lookup_dict, tablefmt, 'trackers', link_fmt=link_fmt)
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


def _help_hpo_samplers(tablefmt: str, link_fmt: Optional[str] = None):
    lines = _get_lines_alternative(
        tablefmt, sampler_resolver.lookup_dict, 'optuna.samplers', 'pykeen.hpo.samplers', link_fmt=link_fmt,
    )
    return tabulate(
        lines,
        headers=['Name', 'Reference', 'Description'],
        tablefmt=tablefmt,
    )


def _get_metrics_lines(tablefmt: str, link_fmt=None):
    if tablefmt == 'rst':
        for name, value in metric_resolver.lookup_dict.items():
            yield name, f':class:`pykeen.evaluation.{value.__name__}`'
    else:
        for field, name, value in get_metric_list():
            if field.name in {'rank_std', 'rank_var', 'rank_mad'}:
                continue
            if tablefmt == 'github':
                yield field.metadata['name'], field.metadata['doc']
            else:
                yield field.metadata['name'], field.metadata['doc'], name, f'pykeen.evaluation.{value.__name__}'


def _get_lines(d, tablefmt, submodule, link_fmt: Optional[str] = None):
    for name, value in sorted(d.items()):
        if tablefmt == 'rst':
            if isinstance(value, type):
                reference = f':class:`pykeen.{submodule}.{value.__name__}`'
            else:
                reference = f':class:`pykeen.{submodule}.{name}`'

            yield name, reference
        elif tablefmt == 'github':
            try:
                ref = value.__name__
                doc = value.__doc__.splitlines()[0]
            except AttributeError:
                ref = name
                doc = value.__class__.__doc__

            reference = f'pykeen.{submodule}.{ref}'
            if link_fmt:
                reference = f'[`{reference}`]({link_fmt.format(reference)})'
            else:
                reference = f'`{reference}`'

            yield name, reference, doc
        else:
            yield name, value.__doc__.splitlines()[0]


def _get_dataset_lines(tablefmt, link_fmt: Optional[str] = None):
    for name, value in sorted(dataset_resolver.lookup_dict.items()):
        reference = f'pykeen.datasets.{value.__name__}'
        if tablefmt == 'rst':
            reference = f':class:`{reference}`'
        elif link_fmt is not None:
            reference = f'[`{reference}`]({link_fmt.format(reference)})'
        else:
            reference = f'`{reference}`'

        try:
            docdata = value.__docdata__
        except AttributeError:
            yield name, reference, '', '', '', ''
            continue

        name = docdata['name']
        statistics = docdata['statistics']
        entities = statistics['entities']
        relations = statistics['relations']
        triples = statistics['triples']

        citation_str = ''
        citation = docdata.get('citation')
        if citation is not None:
            author = citation and citation.get('author')
            year = citation and citation.get('year')
            link = citation and citation.get('link')
            github = citation and citation.get('github')
            if author and year and link:
                _citation_txt = f'{author.capitalize()} *et al*., {year}'
                citation_str = _link(_citation_txt, link, tablefmt)
            elif github:
                link = f'https://github.com/{github}'
                citation_str = _link(github if tablefmt == 'rst' else f'`{github}`', link, tablefmt)
        yield name, reference, citation_str, entities, relations, triples


def _link(text: str, link: str, fmt: str) -> str:
    if fmt == 'rst':
        return f'`{text} <{link}>`_'
    else:
        return f'[{text}]({link})'


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
    loader = FileSystemLoader(HERE.joinpath('templates'))
    environment = Environment(
        autoescape=True,
        loader=loader,
        trim_blocks=False,
    )
    readme_template = environment.get_template('README.md')
    tablefmt = 'github'
    return readme_template.render(
        models=_help_models(tablefmt, link_fmt='https://pykeen.readthedocs.io/en/latest/api/{}.html'),
        n_models=len(model_resolver.lookup_dict),
        regularizers=_help_regularizers(tablefmt, link_fmt='https://pykeen.readthedocs.io/en/latest/api/{}.html'),
        n_regularizers=len(regularizer_resolver.lookup_dict),
        losses=_help_losses(tablefmt, link_fmt='https://pykeen.readthedocs.io/en/latest/api/{}.html'),
        n_losses=len(loss_resolver.lookup_dict),
        datasets=_help_datasets(tablefmt, link_fmt='https://pykeen.readthedocs.io/en/latest/api/{}.html'),
        n_datasets=len(dataset_resolver.lookup_dict),
        training_loops=_help_training(
            tablefmt, link_fmt='https://pykeen.readthedocs.io/en/latest/reference/training.html#{}',
        ),
        n_training_loops=len(training_loop_resolver.lookup_dict),
        negative_samplers=_help_negative_samplers(
            tablefmt, link_fmt='https://pykeen.readthedocs.io/en/latest/api/{}.html',
        ),
        n_negative_samplers=len(negative_sampler_resolver.lookup_dict),
        optimizers=_help_optimizers(tablefmt, link_fmt='https://pytorch.org/docs/stable/optim.html#{}'),
        n_optimizers=len(optimizer_resolver.lookup_dict),
        stoppers=_help_stoppers(
            tablefmt, link_fmt='https://pykeen.readthedocs.io/en/latest/reference/stoppers.html#{}',
        ),
        n_stoppers=len(stopper_resolver.lookup_dict),
        evaluators=_help_evaluators(tablefmt, link_fmt='https://pykeen.readthedocs.io/en/latest/api/{}.html'),
        n_evaluators=len(evaluator_resolver.lookup_dict),
        metrics=_help_metrics(tablefmt, link_fmt='https://pykeen.readthedocs.io/en/latest/api/{}.html'),
        n_metrics=len(get_metric_list()),
        trackers=_help_trackers(tablefmt, link_fmt='https://pykeen.readthedocs.io/en/latest/api/{}.html'),
        n_trackers=len(tracker_resolver.lookup_dict),
        hpo_samplers=_help_hpo_samplers(
            tablefmt, link_fmt='https://optuna.readthedocs.io/en/stable/reference/generated/{}.html',
        ),
        n_hpo_samplers=len(sampler_resolver.lookup_dict),
    )


@main.group()
@click.pass_context
def train(ctx):
    """Train a KGE model."""


for cls in model_resolver.lookup_dict.values():
    train.add_command(build_cli_from_cls(cls))

# Add HPO command
main.add_command(optimize)
main.add_command(experiments)

if __name__ == '__main__':
    main()
