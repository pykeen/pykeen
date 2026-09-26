"""The Hetionet dataset.

Get a summary with ``python -m pykeen.datasets.hetionet``
"""

from typing import Unpack

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import SingleTabbedDataset, SingleTabbedDatasetKwargs

__all__ = [
    "Hetionet",
]

URL = "https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz"


@parse_docdata
class Hetionet(SingleTabbedDataset):
    """The Hetionet dataset from [himmelstein2017]_.

    In its publication [himmelstein2017]_, it is demonstrated to be useful for link prediction in drug repositioning
    and made publicly available through its `GitHub repository <https://github.com/hetio/hetionet>`_ in several formats.
    The link prediction algorithm showcased does not rely on embeddings, which leaves room for interesting comparison.
    One such comparison was made during the master's thesis of Lingling Xu [xu2019]_.
    ---
    name: Hetionet
    citation:
        author: Himmelstein
        year: 2017
        link: https://doi.org/10.7554/eLife.26726
        github: hetio/hetionet
    single: true
    statistics:
        entities: 45158
        relations: 24
        triples: 2250197
        training: 1800157
        testing: 225020
        validation: 225020
    """

    def __init__(self, **kwargs: Unpack[SingleTabbedDatasetKwargs]) -> None:
        """Initialize the `Hetionet <https://github.com/hetio/hetionet>`_ dataset from [himmelstein2017]_."""
        kwargs.setdefault("random_state", 0)
        super().__init__(url=URL, **kwargs)


@click.command()
@verbose_option
def _main() -> None:
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=Hetionet)
    ds.summarize()


if __name__ == "__main__":
    _main()
