# -*- coding: utf-8 -*-

"""Get triples from the UMLS data set."""

import pathlib

from ..base import PathDataSet

__all__ = [
    'UMLS_TRAIN_PATH',
    'UMLS_TEST_PATH',
    'UMLS_VALIDATE_PATH',
    'UMLS',
]

HERE = pathlib.Path(__file__).parent

UMLS_TRAIN_PATH = HERE / 'train.txt'
UMLS_TEST_PATH = HERE / 'test.txt'
UMLS_VALIDATE_PATH = HERE / 'valid.txt'


class UMLS(PathDataSet):
    """The UMLS data set."""

    def __init__(self, **kwargs):
        super().__init__(
            training_path=UMLS_TRAIN_PATH,
            testing_path=UMLS_TEST_PATH,
            validation_path=UMLS_VALIDATE_PATH,
            **kwargs,
        )
