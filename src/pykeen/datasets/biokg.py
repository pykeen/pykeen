"""The `BioKG <https://github.com/dsi-bdi/biokg/>`_ dataset.

Get a summary with ``python -m pykeen.datasets.biokg``.
"""

from typing import Unpack

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import CompressedSingleDatasetKwargs, ZipSingleDataset

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

    def __init__(self, **kwargs: Unpack[CompressedSingleDatasetKwargs]) -> None:
        """Initialize the BioKG dataset from [walsh2020]_."""
        kwargs.setdefault("random_state", 0)
        super().__init__(url=URL, relative_path="biokg.links.tsv", **kwargs)


@click.command()
@verbose_option
def _main() -> None:
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=BioKG)
    ds.summarize()


if __name__ == "__main__":
    _main()
