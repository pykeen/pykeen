# -*- coding: utf-8 -*-

"""WordNet datasets."""

import os
from typing import Optional

from .base import TarFileRemoteDataSet

__all__ = [
    'WN18',
    'WN18RR',
]


class WN18(TarFileRemoteDataSet):
    """The WN18 data set."""

    def __init__(self, cache_root: Optional[str] = None, **kwargs):
        super().__init__(
            url='https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz',
            relative_training_path=os.path.join('wordnet-mlj12', 'wordnet-mlj12-train.txt'),
            relative_testing_path=os.path.join('wordnet-mlj12', 'wordnet-mlj12-test.txt'),
            relative_validation_path=os.path.join('wordnet-mlj12', 'wordnet-mlj12-valid.txt'),
            cache_root=cache_root,
            **kwargs,
        )


class WN18RR(TarFileRemoteDataSet):
    """The WN18-RR data set."""

    def __init__(self, cache_root: Optional[str] = None, **kwargs):
        super().__init__(
            url='https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz',
            relative_training_path='train.txt',
            relative_testing_path='test.txt',
            relative_validation_path='valid.txt',
            cache_root=cache_root,
            **kwargs,
        )
