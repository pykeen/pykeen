# -*- coding: utf-8 -*-

"""Get triples from the Nations dataset."""

import os

from docdata import parse_docdata

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


@parse_docdata
class Nations(PathDataset):
    """The Nations dataset.

    ---
    name: Nations
    statistics:
        entities: 14
        relations: 55
        training: 1592
        testing: 201
        validation: 199
        triples: 1992
    citation:
        author: Zhenfeng Lei
        year: 2017
        github: ZhenfengLei/KGDatasets
    """

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


if __name__ == '__main__':
    Nations().summarize()
