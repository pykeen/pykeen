# -*- coding: utf-8 -*-

"""The `BioKG <https://github.com/dsi-bdi/biokg/>`_ dataset.

Get a summary with ``python -m pykeen.datasets.biokg``
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import SingleTabbedDataset
from ..typing import TorchRandomHint

__all__ = [
    'BioKG',
]

URL = 'https://github.com/dsi-bdi/biokg/releases/download/v1.0.0/biokg.zip'


@parse_docdata
class BioKG(SingleTabbedDataset):
    """The BioKG dataset.

    ---
    name: BioKG
    citation:
        github: dsi-bdi/biokg
    single: true
    statistics:
        entities: 105524
        types: 13
        relations: 17
        triples: 105524
    """

    def __init__(
        self,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        **kwargs,
    ):
        """Initialize the `BioKG <https://github.com/dsi-bdi/biokg/>`_ dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: The random seed to use in splitting the dataset. Defaults to 0.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileSingleDataset`.
        """
        super().__init__(
            url=URL,
            name='biokg.links.tsv',
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    ds = BioKG()
    ds.summarize()


if __name__ == '__main__':
    _main()
