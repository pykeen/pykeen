# -*- coding: utf-8 -*-

"""Get triples from the UMLS dataset."""

import pathlib

from docdata import parse_docdata

from ..base import PathDataset

__all__ = [
    "UMLS_TRAIN_PATH",
    "UMLS_TEST_PATH",
    "UMLS_VALIDATE_PATH",
    "UMLS",
]

HERE = pathlib.Path(__file__).resolve().parent

UMLS_TRAIN_PATH = HERE.joinpath("train.txt")
UMLS_TEST_PATH = HERE.joinpath("test.txt")
UMLS_VALIDATE_PATH = HERE.joinpath("valid.txt")


@parse_docdata
class UMLS(PathDataset):
    """The UMLS dataset.

    ---
    name: Unified Medical Language System
    statistics:
        entities: 135
        relations: 46
        training: 5216
        testing: 661
        validation: 652
        triples: 6529
    citation:
        author: Zhenfeng Lei
        year: 2017
        github: ZhenfengLei/KGDatasets
    """

    def __init__(self, **kwargs):
        """Initialize the UMLS dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=UMLS_TRAIN_PATH,
            testing_path=UMLS_TEST_PATH,
            validation_path=UMLS_VALIDATE_PATH,
            **kwargs,
        )


if __name__ == "__main__":
    UMLS().summarize()
