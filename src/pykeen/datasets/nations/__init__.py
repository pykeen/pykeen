# -*- coding: utf-8 -*-

"""Get triples from the Nations dataset."""

import os

from ..base import PathDataset
from ..literal_base import NumericPathDataset

__all__ = [
    'NATIONS_TRAIN_PATH',
    'NATIONS_TEST_PATH',
    'NATIONS_VALIDATE_PATH',
    'NATIONS_LITERALS_PATH',
    'Nations',
    'NationsLiteral',
]

HERE = os.path.abspath(os.path.dirname(__file__))

NATIONS_TRAIN_PATH = os.path.join(HERE, 'train.txt')
NATIONS_TEST_PATH = os.path.join(HERE, 'test.txt')
NATIONS_VALIDATE_PATH = os.path.join(HERE, 'valid.txt')
NATIONS_LITERALS_PATH = os.path.join(HERE, 'literals.txt')


class Nations(PathDataset):
    """The Nations dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the Nations dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=NATIONS_TRAIN_PATH,
            testing_path=NATIONS_TEST_PATH,
            validation_path=NATIONS_VALIDATE_PATH,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


class NationsLiteral(NumericPathDataset):
    """The Nations dataset with literals."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the Nations dataset with literals.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=NATIONS_TRAIN_PATH,
            testing_path=NATIONS_TEST_PATH,
            validation_path=NATIONS_VALIDATE_PATH,
            literals_path=NATIONS_LITERALS_PATH,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )
