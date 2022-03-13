# -*- coding: utf-8 -*-

"""The `PharmKG <https://github.com/biomed-AI/PharmKG/>`_ datasets.

Get a summary with ``python -m pykeen.datasets.pharmkg``.
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import SingleTabbedDataset, UnpackedRemoteDataset
from ..typing import TorchRandomHint

__all__ = [
    "PharmKG8k",
    "PharmKG",
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

    def __init__(
        self,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        **kwargs,
    ):
        """Initialize the PharmKG dataset from [zheng2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: An optional random state to make the training/testing/validation split reproducible.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            url=RAW_URL,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            read_csv_kwargs=dict(
                usecols=["Entity1_name", "relationship_type", "Entity2_name"],
                sep=",",
            ),
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    from pykeen.datasets import get_dataset

    for cls in [PharmKG8k, PharmKG]:
        get_dataset(dataset=cls).summarize()


if __name__ == "__main__":
    _main()
