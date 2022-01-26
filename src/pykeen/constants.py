# -*- coding: utf-8 -*-

"""Constants for PyKEEN."""

from pathlib import Path
from typing import Tuple

import pystow
import torch

from .typing import Side

__all__ = [
    "PYKEEN_HOME",
    "PYKEEN_DATASETS",
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

# constants for sides
COLUMN_HEAD = 0
COLUMN_RELATION = 1
COLUMN_TAIL = 2
LABEL_HEAD: Side = "head"
LABEL_RELATION: Side = "relation"
LABEL_TAIL: Side = "tail"
# TODO: extend to relation, cf. https://github.com/pykeen/pykeen/pull/728
SIDES: Tuple[Side, ...] = (LABEL_HEAD, LABEL_TAIL)
PART_TO_COLUMN = {
    LABEL_HEAD: COLUMN_HEAD,
    LABEL_RELATION: COLUMN_RELATION,
    LABEL_TAIL: COLUMN_TAIL,
}
