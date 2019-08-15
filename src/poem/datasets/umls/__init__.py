# -*- coding: utf-8 -*-

"""Get triples from the UMLS data set."""

import os

from ...instance_creation_factories import TriplesFactory

__all__ = [
    'TRAIN_PATH',
    'TEST_PATH',
    'VALIDATE_PATH',
    'UmlsTestingTriplesFactory',
    'UmlsTrainingTriplesFactory',
    'UmlsValidationTriplesFactory',
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
