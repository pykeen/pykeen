# -*- coding: utf-8 -*-

"""The OpenBioLink dataset.

Get a summary with ``python -m pykeen.datasets.openbiolink``
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import PackedZipRemoteDataset

__all__ = [
    "OpenBioLink",
    "OpenBioLinkLQ",
]

HQ_URL = "https://samwald.info/res/OpenBioLink_2020_final/HQ_DIR.zip"
LQ_URL = "https://samwald.info/res/OpenBioLink_2020_final/ALL_DIR.zip"


@parse_docdata
class OpenBioLink(PackedZipRemoteDataset):
    """The OpenBioLink dataset.

    OpenBioLink is an open-source, reproducible framework for generating biological
    knowledge graphs for benchmarking link prediction. It is available on GitHub
    at https://github.com/openbiolink/openbiolink and published in [breit2020]_. There are four
    available datasets - this class represents the high quality, directed set.

    ---
    name: OpenBioLink
    citation:
        author: Breit
        year: 2020
        link: https://doi.org/10.1093/bioinformatics/btaa274
        github: openbiolink/openbiolink
    statistics:
        entities: 180992
        relations: 28
        training: 4192002
        testing: 183011
        validation: 188394
        triples: 4563407
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the OpenBioLink dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PackedZipRemoteDataset`.
        """
        super().__init__(
            url=HQ_URL,
            name="HQ_DIR.zip",
            relative_training_path="HQ_DIR/train_test_data/train_sample.csv",
            relative_testing_path="HQ_DIR/train_test_data/test_sample.csv",
            relative_validation_path="HQ_DIR/train_test_data/val_sample.csv",
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


@parse_docdata
class OpenBioLinkLQ(PackedZipRemoteDataset):
    """The low-quality variant of the OpenBioLink dataset.

    ---
    name: OpenBioLink LQ
    citation:
        author: Breit
        year: 2020
        link: https://doi.org/10.1093/bioinformatics/btaa274
        github: openbiolink/openbiolink
    statistics:
        entities: 480876
        relations: 32
        training: 25508954
        testing: 679934
        validation: 1132001
        triples: 27320889
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the OpenBioLink (low quality) dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PackedZipRemoteDataset`.
        """
        super().__init__(
            url=LQ_URL,
            name="ALL_DIR.zip",
            relative_training_path="ALL_DIR/train_test_data/train_sample.csv",
            relative_testing_path="ALL_DIR/train_test_data/test_sample.csv",
            relative_validation_path="ALL_DIR/train_test_data/val_sample.csv",
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    for cls in [OpenBioLink, OpenBioLinkLQ]:
        cls().summarize()


if __name__ == "__main__":
    _main()
