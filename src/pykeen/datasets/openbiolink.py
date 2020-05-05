# -*- coding: utf-8 -*-

"""The OpenBioLink dataset.

Get a summary with ``python -m pykeen.datasets.openbiolink``
"""

import logging

import click

from .dataset import PackedZipRemoteDataSet

__all__ = [
    'OpenBioLink',
    'OpenBioLinkLQ',
]

HQ_URL = 'https://samwald.info/res/OpenBioLink_2020_final/HQ_DIR.zip'
LQ_URL = 'https://samwald.info/res/OpenBioLink_2020_final/ALL_DIR.zip'


class OpenBioLink(PackedZipRemoteDataSet):
    """The OpenBioLink dataset.

    OpenBioLink is an open-source, reproducible framework for generating biological
    knowledge graphs for benchmarking link prediction. It is available on GitHub
    at https://github.com/openbiolink/openbiolink and published in [breit2020]_. There are four
    available data sets - this class represents the high quality, directed set.

    .. [breit2020] Breit, A. (2020) `OpenBioLink: A benchmarking framework for large-scale biomedical link
       prediction <https://doi.org/10.1093/bioinformatics/btaa274>`_, *Bioinformatics*
    """

    def __init__(self, create_inverse_triples: bool = False, eager: bool = False):
        super().__init__(
            url=HQ_URL,
            name='HQ_DIR.zip',
            relative_training_path='HQ_DIR/train_test_data/train_sample.csv',
            relative_testing_path='HQ_DIR/train_test_data/test_sample.csv',
            relative_validation_path='HQ_DIR/train_test_data/val_sample.csv',
            eager=eager,
            create_inverse_triples=create_inverse_triples,
        )


class OpenBioLinkLQ(PackedZipRemoteDataSet):
    """The low-quality variant of the OpenBioLink dataset."""

    def __init__(self, create_inverse_triples: bool = False, eager: bool = False):
        super().__init__(
            url=LQ_URL,
            name='ALL_DIR.zip',
            relative_training_path='ALL_DIR/train_test_data/train_sample.csv',
            relative_testing_path='ALL_DIR/train_test_data/test_sample.csv',
            relative_validation_path='ALL_DIR/train_test_data/val_sample.csv',
            eager=eager,
            create_inverse_triples=create_inverse_triples,
        )


@click.command()
def _main():
    ds = OpenBioLink()
    click.echo(ds.summary_str())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _main()
