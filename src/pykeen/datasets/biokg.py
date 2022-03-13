# -*- coding: utf-8 -*-

"""The `BioKG <https://github.com/dsi-bdi/biokg/>`_ dataset.

Get a summary with ``python -m pykeen.datasets.biokg``.
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import ZipSingleDataset
from ..typing import TorchRandomHint

__all__ = [
    "BioKG",
]

URL = "https://github.com/dsi-bdi/biokg/releases/download/v1.0.0/biokg.zip"


@parse_docdata
class BioKG(ZipSingleDataset):
    """The BioKG dataset from [walsh2020]_.

    ---
    name: BioKG
    citation:
        github: dsi-bdi/biokg
        author: Walsh
        year: 2019
        link: https://doi.org/10.1145/3340531.3412776
    single: true
    statistics:
        entities: 105524
        types: 13
        relations: 17
        triples: 2067997
        training: 1654397
        testing: 206800
        validation: 206800
    """

    def __init__(
        self,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        **kwargs,
    ):
        """Initialize the BioKG dataset from [walsh2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: The random seed to use in splitting the dataset. Defaults to 0.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileSingleDataset`.
        """
        super().__init__(
            url=URL,
            relative_path="biokg.links.tsv",
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=BioKG)
    ds.summarize()


if __name__ == "__main__":
    _main()
