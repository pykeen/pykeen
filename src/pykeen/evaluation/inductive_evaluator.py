# -*- coding: utf-8 -*-

"""Inductive evaluator."""

import logging
import timeit
from textwrap import dedent
from typing import Collection, Iterable, List, Mapping, Optional, Union, cast

import numpy as np
import torch
from tqdm.autonotebook import tqdm

from .evaluator import (
    Evaluator,
    MetricResults,
    create_dense_positive_mask_,
    create_sparse_positive_filter_,
    filter_scores_,
    optional_context_manager,
)
from .rank_based_evaluator import RankBasedEvaluator
from ..models import Model
from ..triples.triples_factory import restrict_triples
from ..triples.utils import get_entities, get_relations
from ..typing import MappedTriples, Mode
from ..utils import format_relative_comparison, split_list_in_batches_iter

__all__ = [
    "InductiveEvaluator",
]

logger = logging.getLogger(__name__)


class InductiveEvaluator(RankBasedEvaluator):
    """
    Inductive version of the evaluator. Main differences:
    - Takes the triple factory argument - on which the evaluation will be executed
    - Takes the mode argument which will be sent to the scoring function
    """
