# coding=utf-8
"""
Freebase datasets.

* FB15k
* FB15k-237
"""
import os
import tarfile
import zipfile
from io import BytesIO

from .dataset import RemoteDataSet

__all__ = [
    'fb15k',
    'fb15k237',
]


class FB15k(RemoteDataSet):
    """FB15k dataset."""

    def __init__(self):
        super().__init__(
            url='https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz',
            relative_training_path=os.path.join('FB15k', 'freebase_mtr100_mte100-train.txt'),
            relative_testing_path=os.path.join('FB15k', 'freebase_mtr100_mte100-test.txt'),
            relative_validation_path=os.path.join('FB15k', 'freebase_mtr100_mte100-valid.txt'),
        )

    def _extract(self, archive_file: BytesIO) -> None:  # noqa: D102
        with tarfile.open(fileobj=archive_file) as tf:
            tf.extractall(path=self.cache_root)


class FB15k237(RemoteDataSet):
    """FB15k-237 dataset."""

    def __init__(self):
        super().__init__(
            url='https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip',
            relative_training_path=os.path.join('Release', 'train.txt'),
            relative_testing_path=os.path.join('Release', 'test.txt'),
            relative_validation_path=os.path.join('Release', 'valid.txt'),
        )

    def _extract(self, archive_file: BytesIO) -> None:  # noqa: D102
        with zipfile.ZipFile(file=archive_file) as zf:
            zf.extractall(path=self.cache_root)


fb15k = FB15k()
fb15k237 = FB15k237()
