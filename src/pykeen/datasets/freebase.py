# -*- coding: utf-8 -*-

"""Freebase datasets.

* FB15k
* FB15k-237
* FB15k-237 with numeric literals
"""

import os

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import PackedZipRemoteDataset, TarFileRemoteDataset
from .remote_literal_base import ZipRemoteDatasetWithRemoteLiterals

__all__ = [
    "FB15k",
    "FB15k237",
    "FB15k237WithLiterals",
]

FB15K237_RELATIONAL_TRIPLES_URL = (
    "https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip"  # noqa
)


@parse_docdata
class FB15k(TarFileRemoteDataset):
    """The FB15k dataset.

    ---
    name: FB15k
    statistics:
        entities: 14951
        relations: 1345
        training: 483142
        testing: 59071
        validation: 50000
        triples: 592213
    citation:
        author: Bordes
        year: 2013
        link: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
    """

    def __init__(self, **kwargs):
        """Initialize the FreeBase 15K dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.

        .. warning:: This dataset contains testing leakage. Use :class:`FB15k237` instead.
        """
        super().__init__(
            url="https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz",
            relative_training_path=os.path.join("FB15k", "freebase_mtr100_mte100-train.txt"),
            relative_testing_path=os.path.join("FB15k", "freebase_mtr100_mte100-test.txt"),
            relative_validation_path=os.path.join("FB15k", "freebase_mtr100_mte100-valid.txt"),
            **kwargs,
        )


@parse_docdata
class FB15k237(PackedZipRemoteDataset):
    """The FB15k-237 dataset.

    ---
    name: FB15k-237
    statistics:
        entities: 14505
        relations: 237
        training: 272115
        testing: 20438
        validation: 17526
        triples: 310079
    citation:
        author: Toutanova
        year: 2015
        link: https://www.aclweb.org/anthology/W15-4007/
    """

    def __init__(self, **kwargs):
        """Initialize the FreeBase 15K (237) dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.ZipFileRemoteDataset`.
        """
        super().__init__(
            url=FB15K237_RELATIONAL_TRIPLES_URL,
            relative_training_path=os.path.join("Release", "train.txt"),
            relative_testing_path=os.path.join("Release", "test.txt"),
            relative_validation_path=os.path.join("Release", "valid.txt"),
            **kwargs,
        )


@parse_docdata
class FB15k237WithLiterals(ZipRemoteDatasetWithRemoteLiterals):
    """The FB15k-237 dataset with numeric literals.

    ---
    name: FB15k-237 with numeric literals
    statistics:
        entities: 14505
        relations: 237
        training: 272115
        testing: 20438
        validation: 17526
        triples: 310079
        literal relations: 121
    citation:
        author: Agustinus Kristiadi et al.
        year: 2018
        link: https://arxiv.org/abs/1802.00934
        license: https://github.com/SmartDataAnalytics/LiteralE/blob/0b0c48fd9b74bf000400199610275ea5c159a44c/LICENSE # noqa
    """

    def __init__(self, **kwargs):
        """Initialize the FB15k-237 dataset with literals.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PackedZipRemoteDataset`.
        """
        super().__init__(
            url=FB15K237_RELATIONAL_TRIPLES_URL,
            relative_training_path=os.path.join("Release", "train.txt"),
            relative_testing_path=os.path.join("Release", "test.txt"),
            relative_validation_path=os.path.join("Release", "valid.txt"),
            numeric_triples_url="https://raw.githubusercontent.com/SmartDataAnalytics/LiteralE/0b0c48fd9b74bf000400199610275ea5c159a44c/data/FB15k-237/literals/numerical_literals.txt",  # noqa
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    for cls in [FB15k, FB15k237, FB15k237WithLiterals]:
        cls().summarize()


if __name__ == "__main__":
    _main()
