# -*- coding: utf-8 -*-

"""Utilities for PyKEEN."""

import logging
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union

import mlflow
import numpy
import numpy as np
import torch
from torch import nn

__all__ = [
    'clamp_norm',
    'compact_mapping',
    'l2_regularization',
    'raise_if_not_cuda_oom',
    'resolve_device',
    'slice_triples',
    'slice_doubles',
    'split_list_in_batches_iter',
    'split_list_in_batches',
    'normalize_string',
    'get_cls',
    'get_until_first_blank',
    'flatten_dictionary',
    'ResultTracker',
    'MLFlowResultTracker',
    'get_embedding_in_canonical_shape',
    'set_random_seed',
    'NoRandomSeedNecessary',
    'Result',
]

logger = logging.getLogger(__name__)


def l2_regularization(
    *xs: torch.Tensor,
    normalize: bool = False
) -> torch.Tensor:
    """
    Compute squared L2-regularization term.

    :param xs: a list of torch.Tensor
        The tensors for which to compute the regularization.
    :param normalize:
        Whether to divide the term by the total number of elements in the tensors.

    :return: The sum of squared value across all tensors.
    """
    regularization_term = sum(x.pow(2).sum() for x in xs)

    # Normalize by the number of elements in the tensors for dimensionality-independent weight tuning.
    if normalize:
        regularization_term /= sum(numpy.prod(x.shape) for x in xs)

    return regularization_term


def resolve_device(device: Union[None, str, torch.device] = None) -> torch.device:
    """Resolve a torch.device given a desired device (string)."""
    if device is None or device == 'gpu':
        device = 'cuda'
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
        logger.warning('No cuda devices were available. The model runs on CPU')
    return device


def slice_triples(triples):
    """Get the heads, relations, and tails from a matrix of triples."""
    return (
        triples[:, 0:1],  # heads
        triples[:, 1:2],  # relations
        triples[:, 2:3],  # tails
    )


def slice_doubles(doubles):
    """Get the heads and relations from a matrix of doubles."""
    return (
        doubles[:, 0:1],  # heads
        doubles[:, 1:2],  # relations
    )


X = TypeVar('X')


def split_list_in_batches(input_list: List[X], batch_size: int) -> List[List[X]]:
    """Split a list of instances in batches of size batch_size."""
    return list(split_list_in_batches_iter(input_list=input_list, batch_size=batch_size))


def split_list_in_batches_iter(input_list: List[X], batch_size: int) -> Iterable[List[X]]:
    """Split a list of instances in batches of size batch_size."""
    return (
        input_list[i:i + batch_size]
        for i in range(0, len(input_list), batch_size)
    )


def normalize_string(s: str, *, suffix: Optional[str] = None) -> str:
    """Normalize a string for lookup."""
    s = s.lower().replace('-', '').replace('_', '')
    if suffix is not None and s.endswith(suffix.lower()):
        return s[:-len(suffix)]
    return s


def get_cls(
    query: Union[None, str, Type[X]],
    base: Type[X],
    lookup_dict: Mapping[str, Type[X]],
    default: Optional[Type[X]] = None,
    suffix: Optional[str] = None,
) -> Type[X]:
    """Get a class by string, default, or implementation."""
    if query is None:
        if default is None:
            raise ValueError(f'No default {base.__name__} set')
        return default
    elif not isinstance(query, (str, type)):
        raise TypeError(f'Invalid {base.__name__} type: {type(query)} - {query}')
    elif isinstance(query, str):
        try:
            return lookup_dict[normalize_string(query, suffix=suffix)]
        except KeyError:
            raise ValueError(f'Invalid {base.__name__} name: {query}')
    elif issubclass(query, base):
        return query
    raise TypeError(f'Not subclass of {base.__name__}: {query}')


def get_until_first_blank(s: str) -> str:
    """Recapitulate all lines in the string until the first blank line."""
    lines = list(s.splitlines())
    try:
        m, _ = min(enumerate(lines), key=lambda line: line == '')
    except ValueError:
        return s
    else:
        return ' '.join(
            line.lstrip()
            for line in lines[:m + 2]
        )


def flatten_dictionary(
    dictionary: Dict[str, Any],
    prefix: Optional[str] = None,
    sep: str = '.',
) -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    real_prefix = tuple() if prefix is None else (prefix,)
    partial_result = _flatten_dictionary(dictionary=dictionary, prefix=real_prefix)
    return {sep.join(k): v for k, v in partial_result.items()}


def _flatten_dictionary(
    dictionary: Dict[str, Any],
    prefix: Tuple[str, ...],
) -> Dict[Tuple[str, ...], Any]:
    """Help flatten a nested dictionary."""
    result = {}
    for k, v in dictionary.items():
        new_prefix = prefix + (k,)
        if isinstance(v, dict):
            result.update(_flatten_dictionary(dictionary=v, prefix=new_prefix))
        else:
            result[new_prefix] = v
    return result


class ResultTracker:
    """A class that tracks the results from a pipeline run."""

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a run with an optional name."""

    def log_params(self, params: Dict[str, Any], prefix: Optional[str] = None) -> None:
        """Log parameters to result store."""

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: Optional[str] = None) -> None:
        """Log metrics to result store.

        :param metrics: The metrics to log.
        :param step: An optional step to attach the metrics to (e.g. the epoch).
        :param prefix: An optional prefix to prepend to every key in metrics.
        """

    def end_run(self) -> None:
        """End a run.

        HAS to be called after the experiment is finished.
        """


class MLFlowResultTracker(ResultTracker):
    """A tracker for MLFlow."""

    def __init__(self, tracking_uri: Optional[str] = None):
        if tracking_uri is None:
            tracking_uri = 'localhost:5000'
        mlflow.set_tracking_uri(tracking_uri)

    def start_run(self, run_name: Optional[str] = None) -> None:  # noqa: D102
        mlflow.start_run(run_name=run_name)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        metrics = flatten_dictionary(dictionary=metrics, prefix=prefix)
        mlflow.log_metrics(metrics=metrics, step=step)

    def log_params(self, params: Dict[str, Any], prefix: Optional[str] = None) -> None:  # noqa: D102
        params = flatten_dictionary(dictionary=params, prefix=prefix)
        mlflow.log_params(params=params)

    def end_run(self) -> None:  # noqa: D102
        mlflow.end_run()


def get_embedding_in_canonical_shape(
    embedding: nn.Embedding,
    ind: Optional[torch.LongTensor],
) -> torch.FloatTensor:
    """Get embedding in canonical shape.

    :param embedding: The embedding.
    :param ind: The indices. If None, return all embeddings.

    :return: shape: (batch_size, num_embeddings, d)
    """
    if ind is None:
        e = embedding.weight.unsqueeze(dim=0)
    else:
        e = embedding(ind).unsqueeze(dim=1)
    return e


def clamp_norm(
    x: torch.Tensor,
    maxnorm: float,
    p: Union[str, int] = 'fro',
    dim: Union[None, int, Iterable[int]] = None,
    eps: float = 1.0e-08,
) -> torch.Tensor:
    """Ensure that a tensor's norm does not exceeds some threshold.

    :param x:
        The vector.
    :param maxnorm:
        The maximum norm (>0).
    :param p:
        The norm type.
    :param dim:
        The dimension(s).
    :param eps:
        A small value to avoid division by zero.

    :return:
        A vector with |x| <= max_norm.
    """
    norm = x.norm(p=p, dim=dim, keepdim=True)
    mask = (norm < maxnorm).type_as(x)
    return mask * x + (1 - mask) * (x / norm.clamp_min(eps) * maxnorm)


def set_random_seed(seed: int):
    """Set the random seed on numpy, torch, and python."""
    return (
        np.random.seed(seed=seed),
        torch.manual_seed(seed=seed),
        random.seed(seed),
    )


class NoRandomSeedNecessary:
    """Used in pipeline when random seed is set automatically."""


def all_in_bounds(
    x: torch.Tensor,
    low: Optional[float] = None,
    high: Optional[float] = None,
    a_tol: float = 0.,
) -> bool:
    """Check if tensor values respect lower and upper bound.

    :param x:
        The tensor.
    :param low:
        The lower bound.
    :param high:
        The upper bound.
    :param a_tol:
        Absolute tolerance.

    """
    # lower bound
    if low is not None and (x < low - a_tol).any():
        return False

    # upper bound
    if high is not None and (x > high + a_tol).any():
        return False

    return True


def raise_if_not_cuda_oom(exception: RuntimeError) -> None:
    """Check whether the catched RuntimeError was due to a CUDA OOM, and if not, re-raise it."""
    if 'CUDA out of memory.' not in exception.args[0]:
        raise exception


def compact_mapping(
    mapping: Mapping[X, int]
) -> Tuple[Mapping[X, int], Mapping[int, int]]:
    """Update a mapping (key -> id) such that the IDs range from 0 to len(mappings) - 1.

    :param mapping:
        The mapping to compact.

    :return: A pair (translated, translation)
        where translated is the updated mapping, and translation a dictionary from old to new ids.
    """
    translation = {
        old_id: new_id
        for new_id, old_id in enumerate(sorted(mapping.values()))
    }
    translated = {
        k: translation[v]
        for k, v in mapping.items()
    }
    return translated, translation


class Result:
    """A superclass of results that can be saved to a directory."""

    def save_to_directory(self, directory: str) -> None:
        """Save the results to the directory."""
        raise NotImplementedError
