# -*- coding: utf-8 -*-

"""YAGO3 datasets."""

import pathlib

import click
from docdata import parse_docdata
from more_click import verbose_option

from pykeen.datasets.remote_literal_base import TarRemoteDatasetWithRemoteLiterals

from .base import TarFileRemoteDataset

__all__ = [
    "YAGO310",
    "YAGO310WithLiterals",
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

    def __init__(self, **kwargs):
        """Initialize the YAGO3-10 dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.
        """
        super().__init__(
            url="https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz",
            relative_training_path=pathlib.PurePath("train.txt"),
            relative_testing_path=pathlib.PurePath("test.txt"),
            relative_validation_path=pathlib.PurePath("valid.txt"),
            **kwargs,
        )


@parse_docdata
class YAGO310WithLiterals(TarRemoteDatasetWithRemoteLiterals):
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
    citations:
        train, validation and test triples:
            author: Mahdisoltani
            year: 2015
            link: http://service.tsi.telecom-paristech.fr/cgi-bin//valipub_download.cgi?dId=284
        literal triples:
            authors: Agustinus Kristiadi, Mohammad Asif Khan, Denis Lukovnikov, Jens Lehmann, Asja Fischer
            year: 2018
            link: https://arxiv.org/abs/1802.00934
            license: https://github.com/SmartDataAnalytics/LiteralE/blob/0b0c48fd9b74bf000400199610275ea5c159a44c/LICENSE   # noqa
    """

    def __init__(self, **kwargs):
        """Initialize the YAGO3-10 dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.
        """
        super().__init__(
            url="https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz",
            relative_training_path=pathlib.PurePath("train.txt"),
            relative_testing_path=pathlib.PurePath("test.txt"),
            relative_validation_path=pathlib.PurePath("valid.txt"),
            numeric_triples_url="https://raw.githubusercontent.com/SmartDataAnalytics/LiteralE/master/data/YAGO3-10/literals/numerical_literals.txt",  # noqa
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    for cls in [YAGO310, YAGO310WithLiterals]:
        cls().summarize()


if __name__ == "__main__":
    _main()
