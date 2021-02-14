# -*- coding: utf-8 -*-

"""Freebase datasets.

* FB15k
* FB15k-237
"""

import os

from .base import TarFileRemoteDataset, ZipFileRemoteDataset

__all__ = [
    'FB15k',
    'FB15k237',
]


class FB15k(TarFileRemoteDataset):
    """The FB15k dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the FreeBase 15K dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.

        .. warning:: This dataset contains testing leakage. Use :class:`FB15k237` instead.
        """
        super().__init__(
            url='https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz',
            relative_training_path=os.path.join('FB15k', 'freebase_mtr100_mte100-train.txt'),
            relative_testing_path=os.path.join('FB15k', 'freebase_mtr100_mte100-test.txt'),
            relative_validation_path=os.path.join('FB15k', 'freebase_mtr100_mte100-valid.txt'),
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


class FB15k237(ZipFileRemoteDataset):
    """The FB15k-237 dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the FreeBase 15K (237) dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.ZipFileRemoteDataset`.
        """
        super().__init__(
            url='https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip',
            relative_training_path=os.path.join('Release', 'train.txt'),
            relative_testing_path=os.path.join('Release', 'test.txt'),
            relative_validation_path=os.path.join('Release', 'valid.txt'),
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )
