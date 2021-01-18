# -*- coding: utf-8 -*-

"""Utilities for PyKEEN."""

import ftplib
import json
import logging
import random
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import (
    Any, Callable, Collection, Dict, Generic, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.modules.batchnorm

from .constants import PYKEEN_BENCHMARKS
from .typing import DeviceHint, TorchRandomHint
from .version import get_git_hash

__all__ = [
    'compose',
    'clamp_norm',
    'compact_mapping',
    'ensure_torch_random_state',
    'format_relative_comparison',
    'imag_part',
    'invert_mapping',
    'is_cuda_oom_error',
    'random_non_negative_int',
    'real_part',
    'resolve_device',
    'split_complex',
    'split_list_in_batches_iter',
    'torch_is_in_1d',
    'normalize_string',
    'normalized_lookup',
    'get_cls',
    'get_until_first_blank',
    'flatten_dictionary',
    'set_random_seed',
    'NoRandomSeedNecessary',
    'Result',
    'fix_dataclass_init_docs',
    'get_benchmark',
]

logger = logging.getLogger(__name__)

#: An error that occurs because the input in CUDA is too big. See ConvE for an example.
_CUDNN_ERROR = 'cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.'

_CUDA_OOM_ERROR = 'CUDA out of memory.'


def resolve_device(device: DeviceHint = None) -> torch.device:
    """Resolve a torch.device given a desired device (string)."""
    if device is None or device == 'gpu':
        device = 'cuda'
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
        logger.warning('No cuda devices were available. The model runs on CPU')
    return device


X = TypeVar('X')


def split_list_in_batches_iter(input_list: List[X], batch_size: int) -> Iterable[List[X]]:
    """Split a list of instances in batches of size batch_size."""
    return (
        input_list[i:i + batch_size]
        for i in range(0, len(input_list), batch_size)
    )


def normalize_string(s: str, *, suffix: Optional[str] = None) -> str:
    """Normalize a string for lookup."""
    s = s.lower().replace('-', '').replace('_', '').replace(' ', '')
    if suffix is not None and s.endswith(suffix.lower()):
        return s[:-len(suffix)]
    return s


def normalized_lookup(classes: Iterable[Type[X]]) -> Mapping[str, Type[X]]:
    """Make a normalized lookup dict."""
    return {
        normalize_string(cls.__name__): cls
        for cls in classes
    }


def get_cls(
    query: Union[None, str, Type[X]],
    base: Type[X],
    lookup_dict: Mapping[str, Type[X]],
    lookup_dict_synonyms: Optional[Mapping[str, Type[X]]] = None,
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
        key = normalize_string(query, suffix=suffix)
        if key in lookup_dict:
            return lookup_dict[key]
        if lookup_dict_synonyms is not None and key in lookup_dict_synonyms:
            return lookup_dict_synonyms[key]
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
    return {sep.join(map(str, k)): v for k, v in partial_result.items()}


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


class compose(Generic[X]):  # noqa:N801
    """A class representing the composition of several functions."""

    def __init__(self, *operations: Callable[[X], X]):
        """Initialize the composition with a sequence of operations."""
        self.operations = operations

    def __call__(self, x: X) -> X:
        """Apply the operations in order to the given tensor."""
        for operation in self.operations:
            x = operation(x)
        return x


def set_random_seed(seed: int) -> Tuple[None, torch.Generator, None]:
    """Set the random seed on numpy, torch, and python."""
    np.random.seed(seed=seed)
    generator = torch.manual_seed(seed=seed)
    random.seed(seed)
    return None, generator, None


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


def is_cuda_oom_error(runtime_error: RuntimeError) -> bool:
    """Check whether the caught RuntimeError was due to CUDA being out of memory."""
    return _CUDA_OOM_ERROR in runtime_error.args[0]


def is_cudnn_error(runtime_error: RuntimeError) -> bool:
    """Check whether the caught RuntimeError was due to a CUDNN error."""
    return _CUDNN_ERROR in runtime_error.args[0]


def compact_mapping(
    mapping: Mapping[X, int],
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


class Result(ABC):
    """A superclass of results that can be saved to a directory."""

    @abstractmethod
    def save_to_directory(self, directory: str, **kwargs) -> None:
        """Save the results to the directory."""

    @abstractmethod
    def save_to_ftp(self, directory: str, ftp: ftplib.FTP) -> None:
        """Save the results to the directory in an FTP server."""

    @abstractmethod
    def save_to_s3(self, directory: str, bucket: str, s3=None) -> None:
        """Save all artifacts to the given directory in an S3 Bucket.

        :param directory: The directory in the S3 bucket
        :param bucket: The name of the S3 bucket
        :param s3: A client from :func:`boto3.client`, if already instantiated
        """


def split_complex(
    x: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Split a complex tensor into real and imaginary part."""
    dim = x.shape[-1] // 2
    return x[..., :dim], x[..., dim:]


def real_part(
    x: torch.FloatTensor,
) -> torch.FloatTensor:
    """Get the real part from a complex tensor."""
    dim = x.shape[-1] // 2
    return x[..., :dim]


def imag_part(
    x: torch.FloatTensor,
) -> torch.FloatTensor:
    """Get the imaginary part from a complex tensor."""
    dim = x.shape[-1] // 2
    return x[..., dim:]


def fix_dataclass_init_docs(cls: Type) -> Type:
    """Fix the ``__init__`` documentation for a :class:`dataclasses.dataclass`.

    :param cls: The class whose docstring needs fixing
    :returns: The class that was passed so this function can be used as a decorator

    .. seealso:: https://github.com/agronholm/sphinx-autodoc-typehints/issues/123
    """
    cls.__init__.__qualname__ = f'{cls.__name__}.__init__'
    return cls


def get_benchmark(name: str) -> Path:
    """Get the benchmark directory for this version."""
    rv = PYKEEN_BENCHMARKS / name / get_git_hash()
    rv.mkdir(exist_ok=True, parents=True)
    return rv


def get_model_io(model) -> BytesIO:
    """Get the model as bytes."""
    model_io = BytesIO()
    torch.save(model, model_io)
    model_io.seek(0)
    return model_io


def get_json_bytes_io(obj) -> BytesIO:
    """Get the JSON as bytes."""
    obj_str = json.dumps(obj, indent=2)
    obj_bytes = obj_str.encode('utf-8')
    return BytesIO(obj_bytes)


def get_df_io(df: pd.DataFrame) -> BytesIO:
    """Get the dataframe as bytes."""
    df_io = BytesIO()
    df.to_csv(df_io, sep='\t', index=False)
    df_io.seek(0)
    return df_io


def ensure_ftp_directory(*, ftp: ftplib.FTP, directory: str) -> None:
    """Ensure the directory exists on the FTP server."""
    try:
        ftp.mkd(directory)
    except ftplib.error_perm:
        pass  # its fine...


K = TypeVar("K")
V = TypeVar("V")


def invert_mapping(mapping: Mapping[K, V]) -> Mapping[V, K]:
    """
    Invert a mapping.

    :param mapping:
        The mapping, key -> value.

    :return:
        The inverse mapping, value -> key.
    """
    num_unique_values = len(set(mapping.values()))
    num_keys = len(mapping)
    if num_unique_values < num_keys:
        raise ValueError(f'Mapping is not bijective! Only {num_unique_values}/{num_keys} are unique.')
    return {
        value: key
        for key, value in mapping.items()
    }


def random_non_negative_int() -> int:
    """Generate a random positive integer."""
    sq = np.random.SeedSequence(np.random.randint(0, np.iinfo(np.int_).max))
    return int(sq.generate_state(1)[0])


def ensure_torch_random_state(random_state: TorchRandomHint) -> torch.Generator:
    """Prepare a random state for PyTorch."""
    if random_state is None:
        random_state = random_non_negative_int()
        logger.warning(f'using automatically assigned random_state={random_state}')
    if isinstance(random_state, int):
        random_state = torch.manual_seed(seed=random_state)
    if not isinstance(random_state, torch.Generator):
        raise TypeError
    return random_state


def torch_is_in_1d(
    query_tensor: torch.LongTensor,
    test_tensor: Union[Collection[int], torch.LongTensor],
    max_id: Optional[int] = None,
    invert: bool = False,
) -> torch.BoolTensor:
    """
    Return a boolean mask with Q[i] in T.

    The method guarantees memory complexity of max(size(Q), size(T)) and is thus, memory-wise, superior to naive
    broadcasting.

    :param query_tensor: shape: S
        The query Q.
    :param test_tensor:
        The test set T.
    :param max_id:
        A maximum ID. If not given, will be inferred.
    :param invert:
        Whether to invert the result.

    :return: shape: S
        A boolean mask.
    """
    # normalize input
    if not isinstance(test_tensor, torch.Tensor):
        test_tensor = torch.as_tensor(data=list(test_tensor), dtype=torch.long)
    if max_id is None:
        max_id = max(query_tensor.max(), test_tensor.max()) + 1
    mask = torch.zeros(max_id, dtype=torch.bool)
    mask[test_tensor] = True
    if invert:
        mask = ~mask
    return mask[query_tensor.view(-1)].view(*query_tensor.shape)


def format_relative_comparison(
    part: int,
    total: int,
) -> str:
    """Format a relative comparison."""
    return f"{part}/{total} ({part / total:2.2%})"


def get_batchnorm_modules(module: torch.nn.Module) -> List[torch.nn.Module]:
    """Return all submodules which are batch normalization layers."""
    return [
        submodule
        for submodule in module.modules()
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm)
    ]
