# -*- coding: utf-8 -*-

"""The PrimeKG dataset.

Get a summary with ``python -m pykeen.datasets.primekg``
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import SingleTabbedDataset
from ..typing import TorchRandomHint

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

    def __init__(
        self,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        **kwargs,
    ):
        """Initialize the PrimeKG dataset from [chandak2022]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: The random seed to use in splitting the dataset. Defaults to 0.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.SingleTabbedDataset`.
        """
        super().__init__(
            url=URL,
            name="primekg.csv",
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            download_kwargs=dict(
                backend="requests",
            ),
            read_csv_kwargs=dict(
                usecols=["x_name", "relation", "y_name"],
                sep=",",
            ),
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=PrimeKG)
    ds.summarize()


if __name__ == "__main__":
    _main()
