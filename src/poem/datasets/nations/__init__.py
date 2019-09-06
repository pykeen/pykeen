# -*- coding: utf-8 -*-

"""Get triples from the Nations data set."""

import os

from ..dataset import DataSet
from ...triples import TriplesFactory

__all__ = [
    'TRAIN_PATH',
    'TEST_PATH',
    'VALIDATE_PATH',
    'NationsTestingTriplesFactory',
    'NationsTrainingTriplesFactory',
    'NationsValidationTriplesFactory',
    'Nations',
    'nations',
]

HERE = os.path.abspath(os.path.dirname(__file__))

TRAIN_PATH = os.path.join(HERE, 'train.txt')
TEST_PATH = os.path.join(HERE, 'test.txt')
VALIDATE_PATH = os.path.join(HERE, 'valid.txt')


class NationsTrainingTriplesFactory(TriplesFactory):
    """A factory for the training portion of the Nations data set."""

    def __init__(self, **kwargs):
        super().__init__(path=TRAIN_PATH, **kwargs)


class NationsTestingTriplesFactory(TriplesFactory):
    """A factory for the testing portion of the Nations data set."""

    def __init__(self, **kwargs):
        super().__init__(path=TEST_PATH, **kwargs)


class NationsValidationTriplesFactory(TriplesFactory):
    """A factory for the validation portion of the Nations data set."""

    def __init__(self, **kwargs):
        super().__init__(path=VALIDATE_PATH, **kwargs)


class Nations(DataSet):
    """The nations data set."""

    def __init__(self, **kwargs):
        super().__init__(
            training_path=TRAIN_PATH,
            testing_path=TEST_PATH,
            validation_path=VALIDATE_PATH,
            **kwargs,
        )


nations = Nations()
