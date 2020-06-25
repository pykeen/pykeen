# -*- coding: utf-8 -*-

"""Get triples from the UMLS data set."""

import os

from ..dataset import PathDataSet
from ...triples import TriplesFactory

__all__ = [
    'TRAIN_PATH',
    'TEST_PATH',
    'VALIDATE_PATH',
    'UmlsTestingTriplesFactory',
    'UmlsTrainingTriplesFactory',
    'UmlsValidationTriplesFactory',
    'Umls',
    'umls',
]

HERE = os.path.abspath(os.path.dirname(__file__))

TRAIN_PATH = os.path.join(HERE, 'train.txt')
TEST_PATH = os.path.join(HERE, 'test.txt')
VALIDATE_PATH = os.path.join(HERE, 'valid.txt')


class UmlsTrainingTriplesFactory(TriplesFactory):
    """A factory for the training portion of the UMLS data set."""

    def __init__(self):
        super().__init__(path=TRAIN_PATH)


class UmlsTestingTriplesFactory(TriplesFactory):
    """A factory for the testing portion of the UMLS data set."""

    def __init__(self):
        super().__init__(path=TEST_PATH)


class UmlsValidationTriplesFactory(TriplesFactory):
    """A factory for the validation portion of the UMLS data set."""

    def __init__(self):
        super().__init__(path=VALIDATE_PATH)


class Umls(PathDataSet):
    """The UMLS data set."""

    def __init__(self, **kwargs):
        super().__init__(
            training_path=TRAIN_PATH,
            testing_path=TEST_PATH,
            validation_path=VALIDATE_PATH,
            **kwargs,
        )


umls = Umls()
