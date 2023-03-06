# -*- coding: utf-8 -*-

"""Lookup for metrics."""

# todo: deprecated

from __future__ import annotations

import logging
import warnings
from typing import Any, Mapping, Type

from .evaluator import MetricResults
from .rank_based_evaluator import RankBasedMetricResults
from ..utils import flatten_dictionary

__all__ = [
    "normalize_flattened_metric_results",
]

logger = logging.getLogger(__name__)


def normalize_flattened_metric_results(
    result: Mapping[str, Any], metric_result_cls: Type[MetricResults] | None = None
) -> Mapping[str, Any]:
    """
    Flatten metric result dictionary and normalize metric keys.

    :param result:
        the result dictionary.
    :param metric_result_cls:
        the metric result class providing metric name normalization

    :return:
        the flattened metric results with normalized metric names.
    """
    # normalize keys
    if metric_result_cls is None:
        warnings.warn("Please explicitly provide a metric result class.", category=DeprecationWarning)
        metric_result_cls = RankBasedMetricResults
    # TODO: find a better way to handle this
    flat_result = flatten_dictionary(result)
    result = {}
    for key, value in flat_result.items():
        try:
            normalized_key = metric_result_cls.key_from_string(key)
        except ValueError as error:
            new_key = key.replace("nondeterministic", "").replace("unknown", "").strip(".").replace("..", ".")
            logger.warning(f"Trying to fix malformed key={key} to key={new_key} (error: {error})")
            normalized_key = metric_result_cls.key_from_string(new_key)
        result[metric_result_cls.key_to_string(normalized_key)] = value
    return result
