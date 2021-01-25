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

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the Kinships dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=KINSHIPS_TRAIN_PATH,
            testing_path=KINSHIPS_TEST_PATH,
            validation_path=KINSHIPS_VALIDATE_PATH,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )
