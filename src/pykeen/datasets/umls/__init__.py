# -*- coding: utf-8 -*-

"""Get triples from the UMLS dataset."""

import os

from ..base import PathDataset

__all__ = [
    'UMLS_TRAIN_PATH',
    'UMLS_TEST_PATH',
    'UMLS_VALIDATE_PATH',
    'UMLS',
]

HERE = os.path.abspath(os.path.dirname(__file__))

UMLS_TRAIN_PATH = os.path.join(HERE, 'train.txt')
UMLS_TEST_PATH = os.path.join(HERE, 'test.txt')
UMLS_VALIDATE_PATH = os.path.join(HERE, 'valid.txt')


class UMLS(PathDataset):
    """The UMLS dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            training_path=UMLS_TRAIN_PATH,
            testing_path=UMLS_TEST_PATH,
            validation_path=UMLS_VALIDATE_PATH,
            **kwargs,
        )
