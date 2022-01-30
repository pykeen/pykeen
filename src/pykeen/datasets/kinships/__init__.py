# -*- coding: utf-8 -*-

"""Get triples from the Kinships dataset."""

import pathlib

from docdata import parse_docdata

from ..base import PathDataset

__all__ = [
    "KINSHIPS_TRAIN_PATH",
    "KINSHIPS_TEST_PATH",
    "KINSHIPS_VALIDATE_PATH",
    "Kinships",
]

HERE = pathlib.Path(__file__).resolve().parent

KINSHIPS_TRAIN_PATH = HERE.joinpath("train.txt")
KINSHIPS_TEST_PATH = HERE.joinpath("test.txt")
KINSHIPS_VALIDATE_PATH = HERE.joinpath("valid.txt")


@parse_docdata
class Kinships(PathDataset):
    """The Kinships dataset.

    ---
    name: Kinships
    statistics:
        entities: 104
        relations: 25
        training: 8544
        testing: 1074
        validation: 1068
        triples: 10686
    citation:
        author: Kemp
        year: 2006
        link: https://www.aaai.org/Papers/AAAI/2006/AAAI06-061.pdf
    """

    def __init__(self, **kwargs):
        """Initialize the Kinships dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=KINSHIPS_TRAIN_PATH,
            testing_path=KINSHIPS_TEST_PATH,
            validation_path=KINSHIPS_VALIDATE_PATH,
            **kwargs,
        )


if __name__ == "__main__":
    Kinships().summarize()
