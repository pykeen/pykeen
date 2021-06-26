# -*- coding: utf-8 -*-

"""The DBpedia datasets from [shi2017b]_.

- GitHub Repository: https://github.com/bxshi/ConMask
- Paper: https://arxiv.org/abs/1711.03438
"""

from docdata import parse_docdata

from .base import UnpackedRemoteDataset

__all__ = [
    'DBpedia50',
]

BASE = 'https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50'
TEST_URL = f'{BASE}/test.txt'
TRAIN_URL = f'{BASE}/train.txt'
VALID_URL = f'{BASE}/valid.txt'


@parse_docdata
class DBpedia50(UnpackedRemoteDataset):
    """The DBpedia50 dataset.

    ---
    name: DBpedia50
    citation:
        author: Shi
        year: 2017
        link: https://arxiv.org/abs/1711.03438
    statistics:
        entities: 24624
        relations: 351
        training: 32203
        testing: 2095
        validation: 123
        triples: 34421
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the DBpedia50 small dataset from [shi2017b]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=TRAIN_URL,
            testing_url=TEST_URL,
            validation_url=VALID_URL,
            create_inverse_triples=create_inverse_triples,
            load_triples_kwargs={
                # as pointed out in https://github.com/pykeen/pykeen/issues/275#issuecomment-776412294,
                # the columns are not ordered properly.
                'column_remapping': [0, 2, 1],
            },
            **kwargs,
        )


if __name__ == '__main__':
    DBpedia50.cli()
