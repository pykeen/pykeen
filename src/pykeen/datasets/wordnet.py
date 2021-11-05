# -*- coding: utf-8 -*-

"""WordNet datasets."""

import pathlib

from docdata import parse_docdata

from .base import TarFileRemoteDataset

__all__ = [
    "WN18",
    "WN18RR",
]


@parse_docdata
class WN18(TarFileRemoteDataset):
    """The WN18 dataset.

    ---
    name: WordNet-18
    statistics:
        entities: 40943
        relations: 18
        training: 141442
        testing: 5000
        validation: 5000
        triples: 151442
    citation:
        author: Bordes
        year: 2014
        link: https://arxiv.org/abs/1301.3485
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the WordNet-18 dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.

        .. warning:: This dataset contains testing leakage. Use :class:`WN18RR` instead.
        """
        super().__init__(
            url="https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz",
            relative_training_path=pathlib.PurePath("wordnet-mlj12", "wordnet-mlj12-train.txt"),
            relative_testing_path=pathlib.PurePath("wordnet-mlj12", "wordnet-mlj12-test.txt"),
            relative_validation_path=pathlib.PurePath("wordnet-mlj12", "wordnet-mlj12-valid.txt"),
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


@parse_docdata
class WN18RR(TarFileRemoteDataset):
    """The WN18-RR dataset.

    ---
    name: WordNet-18 (RR)
    statistics:
        entities: 40559
        relations: 11
        training: 86835
        testing: 2924
        validation: 2824
        triples: 92583
    citation:
        author: Toutanova
        year: 2015
        link: https://www.aclweb.org/anthology/W15-4007/
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the WordNet-18 (RR) dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.
        """
        super().__init__(
            url="https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz",
            relative_training_path="train.txt",
            relative_testing_path="test.txt",
            relative_validation_path="valid.txt",
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


def _main():
    for cls in [WN18, WN18RR]:
        cls().summarize()


if __name__ == "__main__":
    _main()
