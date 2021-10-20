# -*- coding: utf-8 -*-

"""The DB100K dataset."""

from docdata import parse_docdata

from .base import UnpackedRemoteDataset

BASE_URL = "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K"

__all__ = [
    "DB100K",
]


@parse_docdata
class DB100K(UnpackedRemoteDataset):
    """The DB100K dataset from [ding2018]_.

    ---
    name: DB100K
    citation:
        author: Ding
        year: 2018
        link: https://arxiv.org/abs/1805.02408
        github: iieir-km/ComplEx-NNE_AER
    statistics:
        entities: 99604
        relations: 470
        training: 597482
        testing: 50000
        validation: 49997
        triples: 697479
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the DB100K small dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=f"{BASE_URL}/_train.txt",
            testing_url=f"{BASE_URL}/_test.txt",
            validation_url=f"{BASE_URL}/_valid.txt",
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


if __name__ == "__main__":
    DB100K.cli()
