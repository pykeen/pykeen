# coding=utf-8
"""WordNet datasets."""
import os
import tarfile
from io import BytesIO

from .dataset import RemoteDataSet

__all__ = [
    'wn18',
    'wn18rr',
]


class WN18(RemoteDataSet):
    """WN18 dataset."""

    def __init__(self):
        super().__init__(
            url='https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz',
            relative_training_path=os.path.join('wordnet-mlj12', 'wordnet-mlj12-train.txt'),
            relative_testing_path=os.path.join('wordnet-mlj12', 'wordnet-mlj12-test.txt'),
            relative_validation_path=os.path.join('wordnet-mlj12', 'wordnet-mlj12-valid.txt'),
        )

    def _extract(self, archive_file: BytesIO):
        with tarfile.open(fileobj=archive_file) as tf:
            tf.extractall(path=self.cache_root)


class WN18RR(RemoteDataSet):
    """WN18-RR dataset."""

    def __init__(self):
        super().__init__(
            url='https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz',
            relative_training_path='train.txt',
            relative_testing_path='test.txt',
            relative_validation_path='valid.txt',
        )

    def _extract(self, archive_file: BytesIO):
        with tarfile.open(fileobj=archive_file) as tf:
            tf.extractall(path=self.cache_root)


wn18 = WN18()
wn18rr = WN18RR()
