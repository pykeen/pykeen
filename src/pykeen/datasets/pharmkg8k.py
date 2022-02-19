# -*- coding: utf-8 -*-

"""The `PharmKG-8k <https://github.com/biomed-AI/PharmKG/>`_ dataset.

Get a summary with ``python -m pykeen.datasets.pharmkg8k``.
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import UnpackedRemoteDataset

__all__ = [
    "PharmKG8k",
]

BASE_URL = "https://raw.githubusercontent.com/biomed-AI/PharmKG/master/data/PharmKG-8k/"
VALID_URL = f"{BASE_URL}/valid.tsv"
TEST_URL = f"{BASE_URL}/test.tsv"
TRAIN_URL = f"{BASE_URL}/train.tsv"


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

    def __init__(
        self,
        create_inverse_triples: bool = False,
        **kwargs,
    ):
        """Initialize the PharmKG8k dataset from [zheng2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=TRAIN_URL,
            testing_url=TEST_URL,
            validation_url=VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    ds = PharmKG8k()
    ds.summarize()


if __name__ == "__main__":
    _main()
