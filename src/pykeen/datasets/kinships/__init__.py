# -*- coding: utf-8 -*-

"""Get triples from the Kinships dataset."""

import os

from ..base import PathDataset

__all__ = [
    'KINSHIPS_TRAIN_PATH',
    'KINSHIPS_TEST_PATH',
    'KINSHIPS_VALIDATE_PATH',
    'Kinships',
]

HERE = os.path.abspath(os.path.dirname(__file__))

KINSHIPS_TRAIN_PATH = os.path.join(HERE, 'train.txt')
KINSHIPS_TEST_PATH = os.path.join(HERE, 'test.txt')
KINSHIPS_VALIDATE_PATH = os.path.join(HERE, 'valid.txt')


class Kinships(PathDataset):
    """The Kinships dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            training_path=KINSHIPS_TRAIN_PATH,
            testing_path=KINSHIPS_TEST_PATH,
            validation_path=KINSHIPS_VALIDATE_PATH,
            **kwargs,
        )
