# -*- coding: utf-8 -*-

"""Run dataset CLI."""

import click
from more_click import verbose_option


@click.command()
@verbose_option
def main():
    """Load all datasets."""
    from . import datasets
    import docdata
    for name, dataset in sorted(
            datasets.items(),
            key=lambda pair: docdata.get_docdata(pair[1])['statistics']['triples'],
    ):
        click.secho(f'Loading {name}', fg='green', bold=True)
        try:
            dataset().summarize(show_examples=None)
        except Exception as e:
            click.secho(f'Failed {name}', fg='red', bold=True)
            click.secho(str(e), fg='red', bold=True)


if __name__ == '__main__':
    main()
