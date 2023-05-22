# -*- coding: utf-8 -*-

"""API call for datasets."""
import logging
from pathlib import Path

import pandas as pd
import pystow

DATASETS = ["icews14"]
DATAFILES = ["train.txt", "test.txt", "valid.txt"]
RAW_GITHUB = "https://raw.githubusercontent.com/"
BASE_URL = RAW_GITHUB + "BorealisAI/de-simple/master/datasets/"

__all__ = ["get_dataset_base_path"]

logger = logging.getLogger(__name__)


def get_dataset_base_path(dataset) -> Path:
    """Download GDELT, ICEWS05-15 and ICEWS14 datasets from the github repo above.
    :param dataset:
        Dataset choice: GDELT, ICEWS05-15 and ICEWS14.
    :return: the path where the data are stored
    """
    if dataset not in DATASETS:
        raise ValueError(f"Dataset {dataset} not found!")
    datasets = pystow.module("pykeen-temporal", "datasets")
    for datafile in [x for x in DATAFILES]:
        logger.info(f"Ensure dataset {dataset}.{datafile}")
        datasets.ensure(dataset, name=datafile, url=BASE_URL + f"{dataset}/{datafile}")
   
    return datasets.base


if __name__ == "__main__":
    get_dataset_base_path()
