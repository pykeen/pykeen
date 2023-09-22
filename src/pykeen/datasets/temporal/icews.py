# -*- coding: utf-8 -*-

"""ICEWS dataset classes, including ICEWS2014, ICEWS05-15."""


import os

import pykeen

from .api import get_dataset_base_path
from .base import TemporalPathDataset


class ICEWS14(TemporalPathDataset):
    """ICEWS14 Dataset."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=get_dataset_base_path("icews14").joinpath("train.txt"),
            testing_path=get_dataset_base_path("icews14").joinpath("test.txt"),
            validation_path=get_dataset_base_path("icews14").joinpath("valid.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )


class SmallSample(TemporalPathDataset):
    """A Small sample dataset."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        package_base = pykeen.__path__[0]
        super().__init__(
            training_path=os.path.join(package_base, "datasets", "temporal", "small_sample", "train.txt"),
            testing_path=os.path.join(package_base, "datasets", "temporal", "small_sample", "train.txt"),
            validation_path=os.path.join(package_base, "datasets", "temporal", "small_sample", "train.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )
