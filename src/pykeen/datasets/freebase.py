# -*- coding: utf-8 -*-

"""Freebase datasets.

* FB15k
* FB15k-237
"""

import os

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import PackedZipRemoteDataset, TarFileRemoteDataset

__all__ = [
    "FB15k",
    "FB15k237",
]


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

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the FreeBase 15K dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.

        .. warning:: This dataset contains testing leakage. Use :class:`FB15k237` instead.
        """
        super().__init__(
            url="https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz",
            relative_training_path=os.path.join("FB15k", "freebase_mtr100_mte100-train.txt"),
            relative_testing_path=os.path.join("FB15k", "freebase_mtr100_mte100-test.txt"),
            relative_validation_path=os.path.join("FB15k", "freebase_mtr100_mte100-valid.txt"),
            create_inverse_triples=create_inverse_triples,
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

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the FreeBase 15K (237) dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.ZipFileRemoteDataset`.
        """
        super().__init__(
            url="https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip",
            relative_training_path=os.path.join("Release", "train.txt"),
            relative_testing_path=os.path.join("Release", "test.txt"),
            relative_validation_path=os.path.join("Release", "valid.txt"),
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    for cls in [FB15k, FB15k237]:
        cls().summarize()


if __name__ == "__main__":
    _main()
