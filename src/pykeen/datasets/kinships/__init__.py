# -*- coding: utf-8 -*-

"""Get triples from the Kinships data set."""

import pathlib

from ..base import PathDataSet

__all__ = [
    'KINSHIPS_TRAIN_PATH',
    'KINSHIPS_TEST_PATH',
    'KINSHIPS_VALIDATE_PATH',
    'Kinships',
]

HERE = pathlib.Path(__file__).parent

KINSHIPS_TRAIN_PATH = HERE / "train.txt"
KINSHIPS_TEST_PATH = HERE / "test.txt"
KINSHIPS_VALIDATE_PATH = HERE / "valid.txt"


class Kinships(PathDataSet):
    """The Kinships data set."""

    def __init__(self, **kwargs):
        super().__init__(
            training_path=KINSHIPS_TRAIN_PATH,
            testing_path=KINSHIPS_TEST_PATH,
            validation_path=KINSHIPS_VALIDATE_PATH,
            **kwargs,
        )
