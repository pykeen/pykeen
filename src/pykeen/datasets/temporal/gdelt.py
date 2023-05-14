# -*- coding: utf-8 -*-

"""GDELT dataset classes."""


from .api import get_dataset_base_path
from .base import TemporalPathDataset

BASE = get_dataset_base_path()


class GDELTm10(TemporalPathDataset):
    """The GDELT_m10 dataset.

    https://github.com/TemporalKGTeam/A_Unified_Framework_of_Temporal_Knowledge_Graph_Models/tree/main/data
    """

    def __init__(self, create_inverse_quadruples: bool = False, **kwargs):
        """Initialize dataset from paths."""
        super().__init__(
            training_path=BASE.joinpath("gdelt_m10").joinpath("train.txt"),
            testing_path=BASE.joinpath("gdelt_m10").joinpath("test.txt"),
            validation_path=BASE.joinpath("gdelt_m10").joinpath("valid.txt"),
            create_inverse_quadruples=create_inverse_quadruples,
            **kwargs,
        )
