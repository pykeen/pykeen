"""The `PharmKG <https://github.com/biomed-AI/PharmKG/>`_ datasets.

Get a summary with ``python -m pykeen.datasets.pharmkg``.
"""

from typing import Unpack

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import (
    SingleTabbedDataset,
    SingleTabbedDatasetKwargs,
    UnpackedRemoteDataset,
    UnpackedRemoteDataSetKwargs,
)

__all__ = [
    "PharmKG",
    "PharmKG8k",
]

BASE_URL = "https://raw.githubusercontent.com/biomed-AI/PharmKG/master/data/PharmKG-8k/"
VALID_URL = f"{BASE_URL}/valid.tsv"
TEST_URL = f"{BASE_URL}/test.tsv"
TRAIN_URL = f"{BASE_URL}/train.tsv"

RAW_URL = "https://zenodo.org/record/4077338/files/raw_PharmKG-180k.zip"


@parse_docdata
class PharmKG8k(UnpackedRemoteDataset):
    """The PharmKG8k dataset from [zheng2020]_.

    ---
    name: PharmKG8k
    citation:
        github: biomed-AI/PharmKG
        author: Zheng
        year: 2020
        link: https://doi.org/10.1093/bib/bbaa344
    single: true
    statistics:
        entities: 7247
        relations: 28
        training: 386768
        testing: 49764
        validation: 49255
        triples: 485787
    """

    def __init__(self, **kwargs: Unpack[UnpackedRemoteDataSetKwargs]) -> None:
        """Initialize the PharmKG8k dataset from [zheng2020]_.

        :param kwargs: keyword arguments passed to :class:`~pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=TRAIN_URL,
            testing_url=TEST_URL,
            validation_url=VALID_URL,
            **kwargs,
        )


@parse_docdata
class PharmKG(SingleTabbedDataset):
    """The PharmKGFull dataset from [zheng2020]_.

    ---
    name: PharmKG
    citation:
        github: biomed-AI/PharmKG
        author: Zheng
        year: 2020
        link: https://doi.org/10.1093/bib/bbaa344
    single: true
    statistics:
        entities: 188296
        relations: 39
        triples: 1093236
        training: 874588
        testing: 109324
        validation: 109324
    """

    def __init__(self, **kwargs: Unpack[SingleTabbedDatasetKwargs]) -> None:
        """Initialize the PharmKG dataset from [zheng2020]_."""
        kwargs.setdefault("random_state", 0)
        kwargs["read_csv_kwargs"] = {
            "usecols": ["Entity1_name", "relationship_type", "Entity2_name"],
            "sep": ",",
        }
        super().__init__(url=RAW_URL, **kwargs)


@click.command()
@verbose_option
def _main() -> None:
    from pykeen.datasets import get_dataset

    for cls in [PharmKG8k, PharmKG]:
        get_dataset(dataset=cls).summarize()  # type:ignore[arg-type]


if __name__ == "__main__":
    _main()
