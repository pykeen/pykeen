"""The `DRKG <https://github.com/gnn4dr/DRKG>`_ dataset.

Get a summary with ``python -m pykeen.datasets.drkg``
"""

from typing import Unpack

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import CompressedSingleDatasetKwargs, TarFileSingleDataset

__all__ = [
    "DRKG",
]

URL = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"


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
        training: 4699405
        testing: 587426
        validation: 587426
    """

    def __init__(self, **kwargs: Unpack[CompressedSingleDatasetKwargs]) -> None:
        """Initialize the `DRKG <https://github.com/gnn4dr/DRKG>`_ dataset."""
        kwargs.setdefault("random_state", 0)
        super().__init__(url=URL, relative_path="drkg.tsv", **kwargs)


@click.command()
@verbose_option
def _main() -> None:
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=DRKG)
    ds.summarize()


if __name__ == "__main__":
    _main()
