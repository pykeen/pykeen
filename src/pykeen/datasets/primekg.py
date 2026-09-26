"""The PrimeKG dataset.

Get a summary with ``python -m pykeen.datasets.primekg``
"""

from typing import Unpack

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import SingleTabbedDataset, SingleTabbedDatasetKwargs

__all__ = [
    "PrimeKG",
]

URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"


@parse_docdata
class PrimeKG(SingleTabbedDataset):
    """The Precision Medicine Knowledge Graph (PrimeKG) dataset from [chandak2022]_.

    ---
    name: PrimeKG
    citation:
        author: Chandak
        year: 2022
        link: https://doi.org/10.1101/2022.05.01.489928
        github: mims-harvard/PrimeKG
    single: true
    statistics:
        entities: 129375
        relations: 30
        triples: 8100498
        training: 6479992
        testing: 809999
        validation: 810000
    """

    def __init__(self, **kwargs: Unpack[SingleTabbedDatasetKwargs]) -> None:
        """Initialize the PrimeKG dataset from [chandak2022]_."""
        kwargs.setdefault("random_state", 0)
        kwargs["download_kwargs"] = {
            "backend": "requests",
        }
        kwargs["name"] = "primekg.csv"
        kwargs["read_csv_kwargs"] = {
            "usecols": ["x_name", "relation", "y_name"],
            "sep": ",",
        }
        super().__init__(url=URL, **kwargs)


@click.command()
@verbose_option
def _main() -> None:
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=PrimeKG)
    ds.summarize()


if __name__ == "__main__":
    _main()
