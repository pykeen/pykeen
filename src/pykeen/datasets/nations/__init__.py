# -*- coding: utf-8 -*-

"""Get triples from the Nations data set."""

import pathlib

from ..base import PathDataSet

__all__ = [
    'NATIONS_TRAIN_PATH',
    'NATIONS_TEST_PATH',
    'NATIONS_VALIDATE_PATH',
    'Nations',
]

HERE = pathlib.Path(__file__).parent

NATIONS_TRAIN_PATH = HERE / 'train.txt'
NATIONS_TEST_PATH = HERE / 'test.txt'
NATIONS_VALIDATE_PATH = HERE / 'valid.txt'


class Nations(PathDataSet):
    """The Nations data set."""

    def __init__(self, **kwargs):
        super().__init__(
            training_path=NATIONS_TRAIN_PATH,
            testing_path=NATIONS_TEST_PATH,
            validation_path=NATIONS_VALIDATE_PATH,
            **kwargs,
        )
