# -*- coding: utf-8 -*-

"""Get triples from the Nations dataset."""

import pathlib

from docdata import parse_docdata

from ..base import PathDataset
from ..literal_base import NumericPathDataset
from ...triples import TriplesNumericLiteralsFactory

__all__ = [
    "NATIONS_TRAIN_PATH",
    "NATIONS_TEST_PATH",
    "NATIONS_VALIDATE_PATH",
    "NATIONS_LITERALS_PATH",
    "Nations",
    "NationsLiteral",
]

HERE = pathlib.Path(__file__).resolve().parent

NATIONS_TRAIN_PATH = HERE.joinpath("train.txt")
NATIONS_TEST_PATH = HERE.joinpath("test.txt")
NATIONS_VALIDATE_PATH = HERE.joinpath("valid.txt")
NATIONS_LITERALS_PATH = HERE.joinpath("literals.txt")


@parse_docdata
class Nations(PathDataset):
    """The Nations dataset.

    ---
    name: Nations
    statistics:
        entities: 14
        relations: 55
        training: 1592
        testing: 201
        validation: 199
        triples: 1992
    citation:
        author: Zhenfeng Lei
        year: 2017
        github: ZhenfengLei/KGDatasets
    """

    def __init__(self, **kwargs):
        """Initialize the Nations dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=NATIONS_TRAIN_PATH,
            testing_path=NATIONS_TEST_PATH,
            validation_path=NATIONS_VALIDATE_PATH,
            **kwargs,
        )


@parse_docdata
class NationsLiteral(NumericPathDataset):
    """The Nations dataset with literals.

    ---
    name: NationsL
    statistics:
        entities: 14
        relations: 55
        training: 1592
        testing: 201
        validation: 199
        triples: 1992
        literal_relations: 2
        literal_triples: 26
    citation:
        author: Hoyt
        year: 2020
        github: pykeen/pykeen
    """

    training: TriplesNumericLiteralsFactory

    def __init__(self, **kwargs):
        """Initialize the Nations dataset with literals.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=NATIONS_TRAIN_PATH,
            testing_path=NATIONS_TEST_PATH,
            validation_path=NATIONS_VALIDATE_PATH,
            literals_path=NATIONS_LITERALS_PATH,
            **kwargs,
        )


if __name__ == "__main__":
    Nations().summarize()
