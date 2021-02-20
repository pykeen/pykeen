# -*- coding: utf-8 -*-

"""The `DRKG <https://github.com/gnn4dr/DRKG>`_ dataset.

Get a summary with ``python -m pykeen.datasets.drkg``
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import TarFileSingleDataset
from ..typing import TorchRandomHint

__all__ = [
    'DRKG',
]

URL = 'https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz'


@parse_docdata
class DRKG(TarFileSingleDataset):
    """The DRKG dataset.

    ---
    name: Drug Repositioning Knowledge Graph
    citation:
        github: gnn4dr/DRKG
    single: true
    statistics:
        entities: 97238
        types: 13
        relations: 107
        triples: 5874257
    """

    def __init__(
        self,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        **kwargs,
    ):
        """Initialize the `DRKG <https://github.com/gnn4dr/DRKG>`_ dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: The random seed to use in splitting the dataset. Defaults to 0.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileSingleDataset`.
        """
        super().__init__(
            url=URL,
            relative_path='drkg.tsv',
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    ds = DRKG()
    ds.summarize()


if __name__ == '__main__':
    _main()
