# -*- coding: utf-8 -*-

"""ICEWS dataset classes, including ICEWS2014, ICEWS05-15."""


import os

import pykeen_temporal

from .api import get_dataset_base_path
from .base import TemporalPathDataset

BASE = get_dataset_base_path()


class ICEWS14(TemporalPathDataset):
    """ICEWS14 Dataset."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=BASE.joinpath("icews14").joinpath("train.txt"),
            testing_path=BASE.joinpath("icews14").joinpath("test.txt"),
            validation_path=BASE.joinpath("icews14").joinpath("valid.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )


class SmallSample(TemporalPathDataset):
    """A Small sample dataset."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        package_base = pykeen_temporal.__path__[0]
        super().__init__(
            training_path=os.path.join(package_base, "datasets", "small_sample", "train.txt"),
            testing_path=os.path.join(package_base, "datasets", "small_sample", "train.txt"),
            validation_path=os.path.join(package_base, "datasets", "small_sample", "train.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )


class ICEWS14Zhang(TemporalPathDataset):
    """ICEWS14 Dataset."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=BASE.joinpath("icews14-zhang").joinpath("train.txt"),
            testing_path=BASE.joinpath("icews14-zhang").joinpath("test.txt"),
            validation_path=BASE.joinpath("icews14-zhang").joinpath("valid.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )


class ICEWS14Filtered(TemporalPathDataset):
    """ICEWS14 Dataset."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=BASE.joinpath("icews14-filtered").joinpath("train.txt"),
            testing_path=BASE.joinpath("icews14-filtered").joinpath("test.txt"),
            validation_path=BASE.joinpath("icews14-filtered").joinpath("valid.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )


class ICEWS14FilteredSmallTest(TemporalPathDataset):
    """ICEWS14 Dataset."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=BASE.joinpath("icews14-filtered-small-test").joinpath("train.txt"),
            testing_path=BASE.joinpath("icews14-filtered-small-test").joinpath("test.txt"),
            validation_path=BASE.joinpath("icews14-filtered-small-test").joinpath("valid.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )


class ICEWS14Small(TemporalPathDataset):
    """A subset of ICEWS14."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=BASE.joinpath("icews14").joinpath("train_sample.txt"),
            testing_path=BASE.joinpath("icews14").joinpath("train_sample.txt"),
            validation_path=BASE.joinpath("icews14").joinpath("train_sample.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )


class ICEWS5to15(TemporalPathDataset):
    """ICEWS5-15 Dataset."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=BASE.joinpath("icews05-15").joinpath("train.txt"),
            testing_path=BASE.joinpath("icews05-15").joinpath("test.txt"),
            validation_path=BASE.joinpath("icews05-15").joinpath("valid.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )


class ICEWS11to14(TemporalPathDataset):
    """ICEWS11-14 Dataset from.

    https://github.com/TemporalKGTeam/A_Unified_Framework_of_Temporal_Knowledge_Graph_Models/tree/main/data
    """

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=BASE.joinpath("icews11-14").joinpath("train.txt"),
            testing_path=BASE.joinpath("icews11-14").joinpath("test.txt"),
            validation_path=BASE.joinpath("icews11-14").joinpath("valid.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )
