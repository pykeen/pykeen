# -*- coding: utf-8 -*-

"""Get triples from the Kinships dataset."""

import os

from docdata import parse_docdata

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


@parse_docdata
class Kinships(PathDataset):
    """The Kinships dataset.

    ---
    name: Kinships
    statistics:
        entities: 104
        relations: 25
        training: 8544
        testing: 1074
        validation: 1068
        triples: 10686
    citation:
        author: Zhenfeng Lei
        year: 2017
        github: ZhenfengLei/KGDatasets
    """

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


if __name__ == '__main__':
    Kinships().summarize()
