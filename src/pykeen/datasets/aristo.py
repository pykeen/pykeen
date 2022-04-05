# -*- coding: utf-8 -*-

"""The Aristo-v4 dataset."""
import pathlib

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import PackedZipRemoteDataset


@parse_docdata
class AristoV4(PackedZipRemoteDataset):
    """The Aristo-v4 dataset from [chen2021].

    .. note::

        The dataset is based on the Aristo tuple KG from https://aclanthology.org/Q17-1017/.

    .. warning::

        While the original dataset is described as having 44,950 entities, after removing the entities not present in
        train, only 42,016 remain. Similarly, only 1,593 relations occur in training (from the original 1,605 ones).
        Consequently, some validation and testing triples are removed (originally: 20,000). Finally, only 242,567 of
        242,594 original training triples are unique.

    ---
    name: Aristo-v4
    statistics:
        entities: 42016
        relations: 1593
        training: 242567
        testing: 18414
        validation: 18444
        triples: 279425
    citation:
        author: Chen
        year: 2021
        link: https://openreview.net/pdf?id=Qa3uS3H7-Le
    """

    def __init__(self, **kwargs):
        """Initialize the Aristo-v4 dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.ZipFileRemoteDataset`.
        """
        super().__init__(
            url="https://zenodo.org/record/5942560/files/aristo-v4.zip",
            relative_training_path=pathlib.PurePath("train"),
            relative_testing_path=pathlib.PurePath("test"),
            relative_validation_path=pathlib.PurePath("valid"),
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    for cls in [AristoV4]:
        cls().summarize()


if __name__ == "__main__":
    _main()
