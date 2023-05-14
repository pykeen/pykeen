# -*- coding: utf-8 -*-

"""API call for datasets."""
import logging
import os
import tarfile
from pathlib import Path

import pandas as pd
import pystow

DATASETS = ["gdelt", "icews05-15", "icews14"]
DATAFILES = ["train.txt", "test.txt", "valid.txt"]
RAW_GITHUB = "https://raw.githubusercontent.com/"
BASE_URL = RAW_GITHUB + "BorealisAI/de-simple/master/datasets/"
DATASETS_ZHANG = ["gdelt_m10", "icews11-14"]
BASE_URL_ZHANG = (
    RAW_GITHUB + "TemporalKGTeam/A_Unified_Framework_of_Temporal_Knowledge_Graph_Models/main/data/"
)
KINSHIPS_BASE = RAW_GITHUB + "pykeen/pykeen/master/src/pykeen/datasets/kinships/"
WN18RR_BASE = "https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz"

__all__ = ["get_dataset_base_path"]

logger = logging.getLogger(__name__)


def get_dataset_base_path() -> Path:
    """Download GDELT, ICEWS05-15 and ICEWS14 datasets from the github repo above.

    :return: the path where the data are stored
    """
    datasets = pystow.module("pykeen-temporal", "datasets")
    for dataset, datafile in [(x, y) for x in DATASETS for y in DATAFILES]:
        logger.info(f"Ensure dataset {dataset}.{datafile}")
        datasets.ensure(dataset, name=datafile, url=BASE_URL + f"{dataset}/{datafile}")
    for dataset, datafile in [(x, y) for x in DATASETS_ZHANG for y in DATAFILES]:
        logger.info(f"Ensure dataset {dataset}.{datafile}")
        datasets.ensure(dataset, name=datafile, url=BASE_URL_ZHANG + f"{dataset}/{datafile}")
    return datasets.base


def get_static_dataset_base_path() -> Path:
    """Download WN18RR and Kinships dataset then insert an extra column of same timestamp.

    :return: the path where the data are stored
    """
    datasets = pystow.module("pykeen-temporal", "datasets")

    for datafile in DATAFILES:
        logger.info(f"Ensure dataset Kinships.{datafile}")
        datasets.ensure("Kinships", name=datafile, url=KINSHIPS_BASE + datafile)
        df = pd.read_csv(
            datasets.base.joinpath("Kinships").joinpath(datafile),
            sep="\t",
            encoding="utf-8",
            dtype=str,
            header=None,
        )
        df = df.assign(time="2000-01-01")
        df.to_csv(
            datasets.base.joinpath("Kinships").joinpath(datafile),
            header=False,
            index=False,
            sep="\t",
        )

    logger.info("Ensure dataset WN18RR")
    datasets.ensure("WN18RR", name="WN18RR.tar.gz", url=WN18RR_BASE)

    if not os.path.exists(datasets.base.joinpath("WN18RR").joinpath("train.txt")):
        with tarfile.open(datasets.base.joinpath("WN18RR").joinpath("WN18RR.tar.gz")) as tf:
            tf.extractall(path=datasets.base.joinpath("WN18RR"))
        for datafile in DATAFILES:
            df = pd.read_csv(
                datasets.base.joinpath("WN18RR").joinpath(datafile),
                sep="\t",
                encoding="utf-8",
                dtype=str,
                header=None,
            )
            df = df.assign(time="2000-01-01")
            df.to_csv(
                datasets.base.joinpath("WN18RR").joinpath(datafile),
                header=False,
                index=False,
                sep="\t",
            )

    return datasets.base


if __name__ == "__main__":
    get_dataset_base_path()
    get_static_dataset_base_path()
