# -*- coding: utf-8 -*-

"""The Countries dataset."""

from docdata import parse_docdata

from .base import UnpackedRemoteDataset

BASE_URL = 'https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/Countries/Countries_S1'

__all__ = [
    'Countries',
]


@parse_docdata
class Countries(UnpackedRemoteDataset):
    """The Countries dataset.

    ---
    name: Countries
    citation:
        author: Zhenfeng Lei
        year: 2017
        github: ZhenfengLei/KGDatasets
    statistics:
        entities: 271
        relations: 2
        training: 1110
        testing: 24
        validation: 24
        triples: 1158
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the Countries small dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        # GitHub's raw.githubusercontent.com service rejects requests that are streamable. This is
        # normally the default for all of PyKEEN's remote datasets, so just switch the default here.
        kwargs.setdefault('stream', False)
        super().__init__(
            training_url=f'{BASE_URL}/train.txt',
            testing_url=f'{BASE_URL}/test.txt',
            validation_url=f'{BASE_URL}/valid.txt',
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


if __name__ == '__main__':
    Countries.cli()
