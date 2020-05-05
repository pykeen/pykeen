# -*- coding: utf-8 -*-

"""Get triples from the Kinships data set."""

import os

from ..dataset import PathDataSet
from ...triples import TriplesFactory

__all__ = [
    'TRAIN_PATH',
    'TEST_PATH',
    'VALIDATE_PATH',
    'KinshipsTestingTriplesFactory',
    'KinshipsTrainingTriplesFactory',
    'KinshipsValidationTriplesFactory',
    'Kinships',
    'kinships',
]

HERE = os.path.abspath(os.path.dirname(__file__))

TRAIN_PATH = os.path.join(HERE, 'train.txt')
TEST_PATH = os.path.join(HERE, 'test.txt')
VALIDATE_PATH = os.path.join(HERE, 'valid.txt')


class KinshipsTrainingTriplesFactory(TriplesFactory):
    """A factory for the training portion of the Kinships data set."""

    def __init__(self):
        super().__init__(path=TRAIN_PATH)


class KinshipsTestingTriplesFactory(TriplesFactory):
    """A factory for the testing portion of the Kinships data set."""

    def __init__(self):
        super().__init__(path=TEST_PATH)


class KinshipsValidationTriplesFactory(TriplesFactory):
    """A factory for the validation portion of the Kinships data set."""

    def __init__(self):
        super().__init__(path=VALIDATE_PATH)


class Kinships(PathDataSet):
    """The Kinships data set."""

    def __init__(self, **kwargs):
        super().__init__(
            training_path=TRAIN_PATH,
            testing_path=TEST_PATH,
            validation_path=VALIDATE_PATH,
            **kwargs,
        )


kinships = Kinships()
