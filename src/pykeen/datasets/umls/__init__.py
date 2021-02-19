# -*- coding: utf-8 -*-

"""Get triples from the UMLS dataset."""

import os

from ..base import PathDataset
from ...utils_docs import with_structured_docstr

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


@with_structured_docstr
class UMLS(PathDataset):
    """The UMLS dataset.

    ---
    name: Unified Medical Language System
    statistics:
        entities: 135
        relations: 46
        triples: 6529
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the UMLS dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=UMLS_TRAIN_PATH,
            testing_path=UMLS_TEST_PATH,
            validation_path=UMLS_VALIDATE_PATH,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )
