# coding=utf-8
"""YAGO3 datasets."""
import tarfile
from io import BytesIO

from poem.datasets.dataset import RemoteDataSet

__all__ = [
    'yago3_10',
]


class YAGO310(RemoteDataSet):
    """YAGO3-10 contains a subset of YAGO3, with only thos entities with at least 10 relations."""

    def __init__(self):
        super().__init__(
            url='https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz',
            relative_training_path='train.txt',
            relative_testing_path='test.txt',
            relative_validation_path='valid.txt',
        )

    def _extract(self, archive_file: BytesIO):  # noqa: D102
        with tarfile.open(fileobj=archive_file) as tf:
            tf.extractall(path=self.cache_root)


yago3_10 = YAGO310()
