# -*- coding: utf-8 -*-

"""YAGO3 datasets."""

import pathlib

from docdata import parse_docdata

from .base import TarFileRemoteDataset

__all__ = [
    "YAGO310",
]


@parse_docdata
class YAGO310(TarFileRemoteDataset):
    """The YAGO3-10 dataset is a subset of YAGO3 that only contains entities with at least 10 relations.

    ---
    name: YAGO3-10
    statistics:
        entities: 123143
        relations: 37
        training: 1079040
        testing: 4982
        validation: 4978
        triples: 1089000
    citation:
        author: Mahdisoltani
        year: 2015
        link: http://service.tsi.telecom-paristech.fr/cgi-bin//valipub_download.cgi?dId=284
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the YAGO3-10 dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.
        """
        super().__init__(
            url="https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz",
            relative_training_path=pathlib.PurePath("train.txt"),
            relative_testing_path=pathlib.PurePath("test.txt"),
            relative_validation_path=pathlib.PurePath("valid.txt"),
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


if __name__ == "__main__":
    YAGO310().summarize()
