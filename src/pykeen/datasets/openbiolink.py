# -*- coding: utf-8 -*-

"""The OpenBioLink dataset.

Get a summary with ``python -m pykeen.datasets.openbiolink``
"""

import logging
import pathlib
from typing import Optional

import click

from .base import PackedZipRemoteDataSet
from ..typing import Path

__all__ = [
    'OpenBioLink',
    'OpenBioLinkF1',
    'OpenBioLinkF2',
    'OpenBioLinkLQ',
]

HQ_URL = 'https://samwald.info/res/OpenBioLink_2020_final/HQ_DIR.zip'
F1_URL = 'https://github.com/PyKEEN/pykeen-openbiolink-benchmark/raw/master/filter_1/openbiolink_f1.zip'
F2_URL = 'https://github.com/PyKEEN/pykeen-openbiolink-benchmark/raw/master/filter_2/openbiolink_f2.zip'
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

    def __init__(self, cache_root: Optional[Path] = None, create_inverse_triples: bool = False, eager: bool = False):
        internal_root = pathlib.PurePath("HQ_DIR", "train_test_data")
        super().__init__(
            url=HQ_URL,
            name='HQ_DIR.zip',
            relative_training_path=internal_root / "train_sample.csv",
            relative_testing_path=internal_root / "test_sample.csv",
            relative_validation_path=internal_root / "val_sample.csv",
            eager=eager,
            cache_root=cache_root,
            create_inverse_triples=create_inverse_triples,
        )


class OpenBioLinkF1(PackedZipRemoteDataSet):
    """The PyKEEN First Filtered OpenBioLink 2020 Dataset."""

    def __init__(self, cache_root: Optional[Path] = None, create_inverse_triples: bool = False, eager: bool = False):
        super().__init__(
            url=F1_URL,
            name='openbiolink_f1.zip',
            relative_training_path=pathlib.PurePath('train.tsv'),
            relative_testing_path=pathlib.PurePath('test.tsv'),
            relative_validation_path=pathlib.PurePath('val.tsv'),
            cache_root=cache_root,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
        )


class OpenBioLinkF2(PackedZipRemoteDataSet):
    """The PyKEEN Second Filtered OpenBioLink 2020 Dataset."""

    def __init__(self, cache_root: Optional[Path] = None, create_inverse_triples: bool = False, eager: bool = False):
        super().__init__(
            url=F2_URL,
            name='openbiolink_f2.zip',
            relative_training_path=pathlib.PurePath('train.tsv'),
            relative_testing_path=pathlib.PurePath('test.tsv'),
            relative_validation_path=pathlib.PurePath('val.tsv'),
            cache_root=cache_root,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
        )


class OpenBioLinkLQ(PackedZipRemoteDataSet):
    """The low-quality variant of the OpenBioLink dataset."""

    def __init__(self, cache_root: Optional[Path] = None, create_inverse_triples: bool = False, eager: bool = False):
        internal_root = pathlib.PurePath("ALL_DIR", "train_test_data")
        super().__init__(
            url=LQ_URL,
            name='ALL_DIR.zip',
            relative_training_path=internal_root / "train_sample.csv",
            relative_testing_path=internal_root / "test_sample.csv",
            relative_validation_path=internal_root / "val_sample.csv",
            cache_root=cache_root,
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
