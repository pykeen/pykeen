"""The Global Biotic Interactions (GloBI) dataset.

Get a summary with ``python -m pykeen.datasets.globi``
"""

from typing import Unpack

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import SingleTabbedDataset, SingleTabbedDatasetKwargs

__all__ = [
    "Globi",
]

URL = "https://zenodo.org/record/5708970/files/interactions.tsv.gz"


@parse_docdata
class Globi(SingleTabbedDataset):
    """The Global Biotic Interactions (GloBI) dataset.

    ---
    name: Global Biotic Interactions
    citation:
        author: Poelen
        year: 2014
        link: https://doi.org/10.1016/j.ecoinf.2014.08.005
    single: true
    statistics:
        entities: 404207
        relations: 39
        triples: 1966385
        training: 1573108
        testing: 196638
        validation: 196639
    """

    def __init__(self, **kwargs: Unpack[SingleTabbedDatasetKwargs]) -> None:
        """Initialize the GloBI dataset."""
        kwargs.setdefault("random_state", 0)
        kwargs["read_csv_kwargs"] = {
            "usecols": ["sourceTaxonId", "interactionTypeName", "targetTaxonId"],
        }
        super().__init__(url=URL, **kwargs)


@click.command()
@verbose_option
def _main() -> None:
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=Globi)
    ds.summarize()


if __name__ == "__main__":
    _main()
