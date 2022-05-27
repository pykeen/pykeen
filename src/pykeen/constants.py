# -*- coding: utf-8 -*-

"""Constants for PyKEEN."""

from pathlib import Path
from typing import Mapping

import pystow
import torch

from .typing import (
    COLUMN_HEAD,
    COLUMN_RELATION,
    COLUMN_TAIL,
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    Target,
    TargetColumn,
)

__all__ = [
    "PYKEEN_HOME",
    "PYKEEN_DATASETS",
    "PYKEEN_DATASETS_MODULE",
    "PYKEEN_BENCHMARKS",
    "PYKEEN_EXPERIMENTS",
    "PYKEEN_CHECKPOINTS",
    "PYKEEN_LOGS",
    "AGGREGATIONS",
]

#: A manager around the PyKEEN data folder. It defaults to ``~/.data/pykeen``.
#  This can be overridden with the envvar ``PYKEEN_HOME``.
#: For more information, see https://github.com/cthoyt/pystow
PYKEEN_MODULE: pystow.Module = pystow.module("pykeen")
#: A path representing the PyKEEN data folder
PYKEEN_HOME: Path = PYKEEN_MODULE.base
#: A subdirectory of the PyKEEN data folder for datasets, defaults to ``~/.data/pykeen/datasets``
PYKEEN_DATASETS: Path = PYKEEN_MODULE.join("datasets")
PYKEEN_DATASETS_MODULE: pystow.Module = PYKEEN_MODULE.submodule("datasets")
#: A subdirectory of the PyKEEN data folder for benchmarks, defaults to ``~/.data/pykeen/benchmarks``
PYKEEN_BENCHMARKS: Path = PYKEEN_MODULE.join("benchmarks")
#: A subdirectory of the PyKEEN data folder for experiments, defaults to ``~/.data/pykeen/experiments``
PYKEEN_EXPERIMENTS: Path = PYKEEN_MODULE.join("experiments")
#: A subdirectory of the PyKEEN data folder for checkpoints, defaults to ``~/.data/pykeen/checkpoints``
PYKEEN_CHECKPOINTS: Path = PYKEEN_MODULE.join("checkpoints")
#: A subdirectory for PyKEEN logs
PYKEEN_LOGS: Path = PYKEEN_MODULE.join("logs")

PYKEEN_DEFAULT_CHECKPOINT = "PyKEEN_just_saved_my_day.pt"

DEFAULT_DROPOUT_HPO_RANGE = dict(type=float, low=0.0, high=0.5, q=0.1)
#: We define the embedding dimensions as a multiple of 16 because it is computational beneficial (on a GPU)
#: see: https://docs.nvidia.com/deeplearning/performance/index.html#optimizing-performance
DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE = dict(type=int, low=16, high=256, q=16)

USER_DEFINED_CODE = "<user defined>"

AGGREGATIONS = {func.__name__: func for func in [torch.sum, torch.max, torch.mean, torch.logsumexp]}

# TODO: extend to relation, cf. https://github.com/pykeen/pykeen/pull/728
# SIDES: Tuple[Target, ...] = (LABEL_HEAD, LABEL_TAIL)
TARGET_TO_INDEX: Mapping[Target, TargetColumn] = {
    LABEL_HEAD: COLUMN_HEAD,
    LABEL_RELATION: COLUMN_RELATION,
    LABEL_TAIL: COLUMN_TAIL,
}
