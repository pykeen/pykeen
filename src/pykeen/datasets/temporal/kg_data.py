# -*- coding: utf-8 -*-

"""Static KG dataset classes with an extra static time column."""


from .api import get_static_dataset_base_path
from .base import TemporalPathDataset

BASE = get_static_dataset_base_path()


class WN18RR(TemporalPathDataset):
    """WN18RR with an extra column of static time."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=BASE.joinpath("WN18RR").joinpath("train.txt"),
            testing_path=BASE.joinpath("WN18RR").joinpath("test.txt"),
            validation_path=BASE.joinpath("WN18RR").joinpath("valid.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )


class Kinships(TemporalPathDataset):
    """Kinship dataset with an extra column of static time."""

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=BASE.joinpath("Kinships").joinpath("train.txt"),
            testing_path=BASE.joinpath("Kinships").joinpath("test.txt"),
            validation_path=BASE.joinpath("Kinships").joinpath("valid.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )
