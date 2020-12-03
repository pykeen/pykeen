# -*- coding: utf-8 -*-

"""Get triples from the Nations dataset."""

import os

from ..base import PathDataset

__all__ = [
    'NATIONS_TRAIN_PATH',
    'NATIONS_TEST_PATH',
    'NATIONS_VALIDATE_PATH',
    'Nations',
]

HERE = os.path.abspath(os.path.dirname(__file__))

NATIONS_TRAIN_PATH = os.path.join(HERE, 'train.txt')
NATIONS_TEST_PATH = os.path.join(HERE, 'test.txt')
NATIONS_VALIDATE_PATH = os.path.join(HERE, 'valid.txt')


class Nations(PathDataset):
    """The Nations dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            training_path=NATIONS_TRAIN_PATH,
            testing_path=NATIONS_TEST_PATH,
            validation_path=NATIONS_VALIDATE_PATH,
            **kwargs,
        )
