# -*- coding: utf-8 -*-

"""The Countries dataset."""

from docdata import parse_docdata

from .base import UnpackedRemoteDataset

BASE_URL = "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/Countries/Countries_S1"

__all__ = [
    "Countries",
]


@parse_docdata
class Countries(UnpackedRemoteDataset):
    """The Countries dataset.

    ---
    name: Countries
    citation:
        author: Bouchard
        year: 2015
        link: https://www.aaai.org/ocs/index.php/SSS/SSS15/paper/view/10257/10026
    statistics:
        entities: 271
        relations: 2
        training: 1110
        testing: 24
        validation: 24
        triples: 1158
    """

    def __init__(self, **kwargs):
        """Initialize the Countries small dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        super().__init__(
            training_url=f"{BASE_URL}/train.txt",
            testing_url=f"{BASE_URL}/test.txt",
            validation_url=f"{BASE_URL}/valid.txt",
            **kwargs,
        )


if __name__ == "__main__":
    Countries.cli()
