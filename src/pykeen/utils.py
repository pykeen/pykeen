"""Utilities for PyKEEN."""

from __future__ import annotations

import ftplib
import functools
import itertools as itt
import json
import logging
import math
import operator
import os
import pathlib
import random
import re
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, MutableMapping, Sequence
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Generic,
    TextIO,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.modules.batchnorm
import yaml
from class_resolver import normalize_string
from docdata import get_docdata
from torch import nn
from typing_extensions import ParamSpec

from .constants import PYKEEN_BENCHMARKS
from .typing import BoolTensor, DeviceHint, FloatTensor, LongTensor, MappedTriples, TorchRandomHint
from .version import get_git_hash

__all__ = [
    "at_least_eps",
    "broadcast_upgrade_to_sequences",
    "compose",
    "clamp_norm",
    "compact_mapping",
    "create_relation_to_entity_set_mapping",
    "ensure_complex",
    "ensure_torch_random_state",
    "format_relative_comparison",
    "invert_mapping",
    "random_non_negative_int",
    "resolve_device",
    "split_complex",
    "normalize_string",
    "get_until_first_blank",
    "flatten_dictionary",
    "set_random_seed",
    "NoRandomSeedNecessary",
    "Result",
    "fix_dataclass_init_docs",
    "get_benchmark",
    "upgrade_to_sequence",
    "ensure_tuple",
    "unpack_singletons",
    "extend_batch",
    "check_shapes",
    "all_in_bounds",
    "view_complex",
    "combine_complex",
    "get_model_io",
    "get_json_bytes_io",
    "get_df_io",
    "ensure_ftp_directory",
    "get_batchnorm_modules",
    "get_dropout_modules",
    "calculate_broadcasted_elementwise_result_shape",
    "estimate_cost_of_sequence",
    "get_optimal_sequence",
    "tensor_sum",
    "tensor_product",
    "negative_norm_of_sum",
    "negative_norm",
    "project_entity",
    "get_expected_norm",
    "Bias",
    "complex_normalize",
    "lp_norm",
    "powersum_norm",
    "get_devices",
    "get_preferred_device",
    "triple_tensor_to_set",
    "is_triple_tensor_subset",
    "logcumsumexp",
    "get_connected_components",
    "normalize_path",
    "get_edge_index",
    "prepare_filter_triples",
    "nested_get",
    "rate_limited",
    "ExtraReprMixin",
    "einsum",
    "isin_many_dim",
    "split_workload",
    "batched_dot",
]

logger = logging.getLogger(__name__)

P = ParamSpec("P")
X = TypeVar("X")


#: An error that occurs because the input in CUDA is too big. See ConvE for an example.
_CUDNN_ERROR = "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input."


def at_least_eps(x: FloatTensor) -> FloatTensor:
    """Make sure a tensor is greater than zero."""
    # get datatype specific epsilon
    eps = torch.finfo(x.dtype).eps
    # clamp minimum value
    return x.clamp(min=eps)


def resolve_device(device: DeviceHint = None) -> torch.device:
    """Resolve a torch.device given a desired device (string)."""
    if device is None or device == "gpu":
        device = "cuda"
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == "cuda":
        device = torch.device("cpu")
        logger.warning("No cuda devices were available. The model runs on CPU")
    return device


class DeviceResolutionError(ValueError):
    """An error in the resolution of a model's device."""


class AmbiguousDeviceError(DeviceResolutionError):
    """An error raised if there is ambiguity in device resolution."""

    def __init__(self, module: nn.Module) -> None:
        """Initialize the error."""
        _info = defaultdict(list)
        for name, tensor in itt.chain(module.named_parameters(), module.named_buffers()):
            _info[tensor.data.device].append(name)
        info = {device: sorted(tensor_names) for device, tensor_names in _info.items()}
        super().__init__(f"Ambiguous device! Found: {list(info.keys())}\n\n{info}")


def get_devices(module: nn.Module) -> Collection[torch.device]:
    """Return the device(s) from each components of the model."""
    return {tensor.data.device for tensor in itt.chain(module.parameters(), module.buffers())}


def get_preferred_device(module: nn.Module, allow_ambiguity: bool = True) -> torch.device:
    """Return the preferred device."""
    devices = get_devices(module=module)
    if len(devices) == 0:
        raise DeviceResolutionError("Could not infer device, since there are neither parameters nor buffers.")
    if len(devices) == 1:
        return next(iter(devices))
    if not allow_ambiguity:
        raise AmbiguousDeviceError(module=module)
    # try to resolve ambiguous device; there has to be at least one cuda device
    cuda_devices = {d for d in devices if d.type == "cuda"}
    if len(cuda_devices) == 1:
        return next(iter(cuda_devices))
    raise AmbiguousDeviceError(module=module)


def get_until_first_blank(s: str) -> str:
    """Recapitulate all lines in the string until the first blank line."""
    lines = list(s.splitlines())
    try:
        m, _ = min(enumerate(lines), key=lambda line: line == "")
    except ValueError:
        return s
    else:
        return " ".join(line.lstrip() for line in lines[: m + 2])


def flatten_dictionary(
    dictionary: Mapping[str, Any],
    prefix: str | None = None,
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten a nested dictionary."""
    real_prefix = tuple() if prefix is None else (prefix,)
    partial_result = _flatten_dictionary(dictionary=dictionary, prefix=real_prefix)
    return {sep.join(map(str, k)): v for k, v in partial_result.items()}


def _flatten_dictionary(
    dictionary: Mapping[str, Any],
    prefix: tuple[str, ...],
) -> dict[tuple[str, ...], Any]:
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
    p: str | int = "fro",
    dim: None | int | Iterable[int] = None,
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

    :return:
        A vector with $|x| <= maxnorm$.
    """
    norm = x.norm(p=p, dim=dim, keepdim=True)
    mask = (norm < maxnorm).type_as(x)
    return mask * x + (1 - mask) * (x / at_least_eps(norm) * maxnorm)


class compose(Generic[X]):  # noqa:N801
    """A class representing the composition of several functions."""

    def __init__(self, *operations: Callable[[X], X], name: str):
        """Initialize the composition with a sequence of operations.

        :param operations: unary operations that will be applied in succession
        :param name: The name of the composed function.
        """
        self.operations = operations
        self.name = name

    @property
    def __name__(self) -> str:
        """Get the name of this composition."""
        return self.name

    def __call__(self, x: X) -> X:
        """Apply the operations in order to the given tensor."""
        for operation in self.operations:
            x = operation(x)
        return x


def set_random_seed(seed: int) -> tuple[None, torch.Generator, None]:
    """Set the random seed on numpy, torch, and python.

    :param seed: The seed that will be used in :func:`np.random.seed`, :func:`torch.manual_seed`,
        and :func:`random.seed`.
    :returns: A three tuple with None, the torch generator, and None.
    """
    np.random.seed(seed=seed)
    generator = torch.manual_seed(seed=seed)
    random.seed(seed)
    return None, generator, None


class NoRandomSeedNecessary:
    """Used in pipeline when random seed is set automatically."""


def all_in_bounds(
    x: torch.Tensor,
    low: float | None = None,
    high: float | None = None,
    a_tol: float = 0.0,
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

    :returns: If all values are within the given bounds
    """
    # lower bound
    if low is not None and (x < low - a_tol).any():
        return False

    # upper bound
    if high is not None and (x > high + a_tol).any():
        return False

    return True


def is_cudnn_error(runtime_error: RuntimeError) -> bool:
    """Check whether the caught RuntimeError was due to a CUDNN error."""
    return _CUDNN_ERROR in runtime_error.args[0]


def compact_mapping(
    mapping: Mapping[X, int],
) -> tuple[Mapping[X, int], Mapping[int, int]]:
    """Update a mapping (key -> id) such that the IDs range from 0 to len(mappings) - 1.

    :param mapping:
        The mapping to compact.

    :return: A pair (translated, translation)
        where translated is the updated mapping, and translation a dictionary from old to new ids.
    """
    translation = {old_id: new_id for new_id, old_id in enumerate(sorted(mapping.values()))}
    translated = {k: translation[v] for k, v in mapping.items()}
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
    x: FloatTensor,
) -> tuple[FloatTensor, FloatTensor]:
    """Split a complex tensor into real and imaginary part."""
    x = torch.view_as_real(x)
    return x[..., 0], x[..., 1]


def view_complex(x: FloatTensor) -> torch.Tensor:
    """Convert a PyKEEN complex tensor representation into a torch one."""
    real, imag = split_complex(x=x)
    return torch.complex(real=real, imag=imag)


def view_complex_native(x: FloatTensor) -> torch.Tensor:
    """Convert a PyKEEN complex tensor representation into a torch one using :func:`torch.view_as_complex`."""
    return torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))


def combine_complex(
    x_re: FloatTensor,
    x_im: FloatTensor,
) -> FloatTensor:
    """Combine a complex tensor from real and imaginary part."""
    return torch.view_as_complex(torch.stack([x_re, x_im], dim=-1))


def fix_dataclass_init_docs(cls: type) -> type:
    """Fix the ``__init__`` documentation for a :class:`dataclasses.dataclass`.

    :param cls: The class whose docstring needs fixing
    :returns: The class that was passed so this function can be used as a decorator

    .. seealso:: https://github.com/agronholm/sphinx-autodoc-typehints/issues/123
    """
    cls.__init__.__qualname__ = f"{cls.__name__}.__init__"  # type:ignore
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
    obj_bytes = obj_str.encode("utf-8")
    return BytesIO(obj_bytes)


def get_df_io(df: pd.DataFrame) -> BytesIO:
    """Get the dataframe as bytes."""
    df_io = BytesIO()
    df.to_csv(df_io, sep="\t", index=False)
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

    :raises ValueError: if the mapping is not bijective
    """
    num_unique_values = len(set(mapping.values()))
    num_keys = len(mapping)
    if num_unique_values < num_keys:
        raise ValueError(f"Mapping is not bijective! Only {num_unique_values}/{num_keys} are unique.")
    return {value: key for key, value in mapping.items()}


def random_non_negative_int() -> int:
    """Generate a random positive integer."""
    rng = np.random.default_rng()
    dtype = np.int32
    max_value = np.iinfo(dtype).max
    return int(rng.integers(max_value, dtype=dtype))


def ensure_torch_random_state(random_state: TorchRandomHint) -> torch.Generator:
    """Prepare a random state for PyTorch."""
    if random_state is None:
        random_state = random_non_negative_int()
        logger.warning(f"using automatically assigned random_state={random_state}")
    if isinstance(random_state, int):
        random_state = torch.manual_seed(seed=random_state)
    if not isinstance(random_state, torch.Generator):
        raise TypeError
    return random_state


def format_relative_comparison(
    part: int,
    total: int,
) -> str:
    """Format a relative comparison."""
    return f"{part}/{total} ({part / total:2.2%})"


def get_batchnorm_modules(module: torch.nn.Module) -> list[torch.nn.Module]:
    """Return all submodules which are batch normalization layers."""
    return [submodule for submodule in module.modules() if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm)]


def get_dropout_modules(module: torch.nn.Module) -> list[torch.nn.Module]:
    """Return all submodules which are dropout layers."""
    return [submodule for submodule in module.modules() if isinstance(submodule, torch.nn.modules.dropout._DropoutNd)]


def calculate_broadcasted_elementwise_result_shape(
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> tuple[int, ...]:
    """Determine the return shape of a broadcasted elementwise operation."""
    return tuple(max(a, b) for a, b in zip(first, second))


def estimate_cost_of_sequence(
    shape: tuple[int, ...],
    *other_shapes: tuple[int, ...],
) -> int:
    """Cost of a sequence of broadcasted element-wise operations of tensors, given their shapes."""
    return sum(
        map(
            np.prod,
            itt.islice(
                itt.accumulate(
                    (shape,) + other_shapes,
                    calculate_broadcasted_elementwise_result_shape,
                ),
                1,
                None,
            ),
        )
    )


@functools.lru_cache(maxsize=32)
def _get_optimal_sequence(
    *sorted_shapes: tuple[int, ...],
) -> tuple[int, tuple[int, ...]]:
    """Find the optimal sequence in which to combine tensors element-wise based on the shapes.

    The shapes should be sorted to enable efficient caching.
    :param sorted_shapes:
        The shapes of the tensors to combine.
    :return:
        The optimal execution order (as indices), and the cost.
    """
    return min(
        (estimate_cost_of_sequence(*(sorted_shapes[i] for i in p)), p)
        for p in itt.permutations(list(range(len(sorted_shapes))))
    )


@functools.lru_cache(maxsize=64)
def get_optimal_sequence(*shapes: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """Find the optimal sequence in which to combine tensors elementwise based on the shapes.

    :param shapes:
        The shapes of the tensors to combine.
    :return:
        The optimal execution order (as indices), and the cost.
    """
    # create sorted list of shapes to allow utilization of lru cache (optimal execution order does not depend on the
    # input sorting, as the order is determined by re-ordering the sequence anyway)
    arg_sort = sorted(range(len(shapes)), key=shapes.__getitem__)

    # Determine optimal order and cost
    cost, optimal_order = _get_optimal_sequence(*(shapes[new_index] for new_index in arg_sort))

    # translate back to original order
    optimal_order = tuple(arg_sort[i] for i in optimal_order)

    return cost, optimal_order


def _reorder(
    tensors: tuple[FloatTensor, ...],
) -> tuple[FloatTensor, ...]:
    """Re-order tensors for broadcasted element-wise combination of tensors.

    The optimal execution plan gets cached so that the optimization is only performed once for a fixed set of shapes.

    :param tensors:
        The tensors, in broadcastable shape.

    :return:
        The re-ordered tensors in optimal processing order.
    """
    if len(tensors) < 3:
        return tensors
    # determine optimal processing order
    shapes = tuple(tuple(t.shape) for t in tensors)
    if len(set(s[0] for s in shapes if s)) < 2:
        # heuristic
        return tensors
    order = get_optimal_sequence(*shapes)[1]
    return tuple(tensors[i] for i in order)


def tensor_sum(*tensors: FloatTensor) -> FloatTensor:
    """Compute element-wise sum of tensors in broadcastable shape."""
    return sum(_reorder(tensors=tensors))


def tensor_product(*tensors: FloatTensor) -> FloatTensor:
    """Compute element-wise product of tensors in broadcastable shape."""
    head, *rest = _reorder(tensors=tensors)
    return functools.reduce(operator.mul, rest, head)


def negative_norm_of_sum(
    *x: FloatTensor,
    p: str | int | float = 2,
    power_norm: bool = False,
) -> FloatTensor:
    """Evaluate negative norm of a sum of vectors on already broadcasted representations.

    :param x: shape: (batch_size, num_heads, num_relations, num_tails, dim)
        The representations.
    :param p:
        The p for the norm. cf. :func:`torch.linalg.vector_norm`.
    :param power_norm:
        Whether to return $|x-y|_p^p$, cf. https://github.com/pytorch/pytorch/issues/28119

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm(tensor_sum(*x), p=p, power_norm=power_norm)


def negative_norm(
    x: FloatTensor,
    p: str | int | float = 2,
    power_norm: bool = False,
) -> FloatTensor:
    """Evaluate negative norm of a vector.

    :param x: shape: (batch_size, num_heads, num_relations, num_tails, dim)
        The vectors.
    :param p:
        The p for the norm. cf. :func:`torch.linalg.vector_norm`.
    :param power_norm:
        Whether to return $|x-y|_p^p$, cf. https://github.com/pytorch/pytorch/issues/28119

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    if power_norm:
        assert not isinstance(p, str)
        return -(x.abs() ** p).sum(dim=-1)

    return -x.norm(p=p, dim=-1)


def project_entity(
    e: FloatTensor,
    e_p: FloatTensor,
    r_p: FloatTensor,
) -> FloatTensor:
    r"""Project entity relation-specific.

    .. math::

        e_{\bot} = M_{re} e
                 = (r_p e_p^T + I^{d_r \times d_e}) e
                 = r_p e_p^T e + I^{d_r \times d_e} e
                 = r_p (e_p^T e) + e'

    and additionally enforces

    .. math::

        \|e_{\bot}\|_2 \leq 1

    :param e: shape: (..., d_e)
        The entity embedding.
    :param e_p: shape: (..., d_e)
        The entity projection.
    :param r_p: shape: (..., d_r)
        The relation projection.

    :return: shape: (..., d_r)

    """
    # The dimensions affected by e'
    change_dim = min(e.shape[-1], r_p.shape[-1])

    # Project entities
    # r_p (e_p.T e) + e'
    e_bot = r_p * torch.sum(e_p * e, dim=-1, keepdim=True)
    e_bot[..., :change_dim] += e[..., :change_dim]

    # Enforce constraints
    e_bot = clamp_norm(e_bot, p=2, dim=-1, maxnorm=1)

    return e_bot


def upgrade_to_sequence(x: X | Sequence[X]) -> Sequence[X]:
    """Ensure that the input is a sequence.

    .. note ::
        While strings are technically also a sequence, i.e.,

        .. code-block:: python

            isinstance("test", typing.Sequence) is True

        this may lead to unexpected behaviour when calling `upgrade_to_sequence("test")`.
        We thus handle strings as non-sequences. To recover the other behavior, the following may be used:

        .. code-block:: python

            upgrade_to_sequence(tuple("test"))


    :param x: A literal or sequence of literals
    :return: If a literal was given, a one element tuple with it in it. Otherwise, return the given value.

    >>> upgrade_to_sequence(1)
    (1,)
    >>> upgrade_to_sequence((1, 2, 3))
    (1, 2, 3)
    >>> upgrade_to_sequence("test")
    ('test',)
    >>> upgrade_to_sequence(tuple("test"))
    ('t', 'e', 's', 't')
    """
    return x if (isinstance(x, Sequence) and not isinstance(x, str)) else (x,)  # type: ignore


def broadcast_upgrade_to_sequences(*xs: X | Sequence[X]) -> Sequence[Sequence[X]]:
    """Apply upgrade_to_sequence to each input, and afterwards repeat singletons to match the maximum length.

    :param xs: length: m
        the inputs.

    :return:
        a sequence of length m, where each element is a sequence and all elements have the same length.

    :raises ValueError:
        if there is a non-singleton sequence input with length different from the maximum sequence length.

    >>> broadcast_upgrade_to_sequences(1)
    ((1,),)
    >>> broadcast_upgrade_to_sequences(1, 2)
    ((1,), (2,))
    >>> broadcast_upgrade_to_sequences(1, (2, 3))
    ((1, 1), (2, 3))
    """
    # upgrade to sequence
    xs_ = [upgrade_to_sequence(x) for x in xs]
    # broadcast
    max_len = max(map(len, xs_))
    for i in range(len(xs_)):
        x = xs_[i]
        if len(x) < max_len:
            if len(x) != 1:
                raise ValueError(f"Length mismatch: maximum length: {max_len}, but encountered length {len(x)}, too.")
            xs_[i] = tuple(list(x) * max_len)
    return tuple(xs_)


def ensure_tuple(*x: X | Sequence[X]) -> Sequence[Sequence[X]]:
    """Ensure that all elements in the sequence are upgraded to sequences.

    :param x: A sequence of sequences or literals
    :return: An upgraded sequence of sequences

    >>> ensure_tuple(1, (1,), (1, 2))
    ((1,), (1,), (1, 2))
    """
    return tuple(upgrade_to_sequence(xx) for xx in x)


def unpack_singletons(*xs: tuple[X]) -> Sequence[X | tuple[X]]:
    """Unpack sequences of length one.

    :param xs: A sequence of tuples of length 1 or more
    :return: An unpacked sequence of sequences

    >>> unpack_singletons((1,), (1, 2), (1, 2, 3))
    (1, (1, 2), (1, 2, 3))
    """
    return tuple(x[0] if len(x) == 1 else x for x in xs)


def extend_batch(
    batch: MappedTriples,
    max_id: int,
    dim: int,
    ids: LongTensor | None = None,
) -> MappedTriples:
    """Extend batch for 1-to-all scoring by explicit enumeration.

    :param batch: shape: (batch_size, 2)
        The batch.
    :param max_id:
        The maximum IDs to enumerate.
    :param ids: shape: (num_ids,) | (batch_size, num_ids)
        explicit IDs
    :param dim: in {0,1,2}
        The column along which to insert the enumerated IDs.

    :return: shape: (batch_size * num_choices, 3)
        A large batch, where every pair from the original batch is combined with every ID.
    """
    # normalize ids: -> ids.shape: (batch_size, num_ids)
    if ids is None:
        ids = torch.arange(max_id, device=batch.device)
    if ids.ndimension() < 2:
        ids = ids.unsqueeze(dim=0)
    assert ids.ndimension() == 2

    # normalize batch -> batch.shape: (batch_size, 1, 3)
    batch = batch.unsqueeze(dim=1)

    # allocate memory
    hrt_batch = batch.new_empty(size=(batch.shape[0], ids.shape[-1], 3))

    # copy ids
    hrt_batch[..., dim] = ids
    hrt_batch[..., [i for i in range(3) if i != dim]] = batch

    # reshape
    return hrt_batch.view(-1, 3)


def check_shapes(
    *x: tuple[torch.Tensor | tuple[int, ...], str],
    raise_on_errors: bool = True,
) -> bool:
    """Verify that a sequence of tensors are of matching shapes.

    :param x:
        A tuple (t, s), where `t` is a tensor, or an actual shape of a tensor (a tuple of integers), and `s` is a
        string, where each character corresponds to a (named) dimension. If the shapes of different tensors share a
        character, the corresponding dimensions are expected to be of equal size.
    :param raise_on_errors:
        Whether to raise an exception in case of a mismatch.

    :return:
        Whether the shapes matched.

    :raises ValueError:
        If the shapes mismatch and raise_on_error is True.

    Examples:
    >>> check_shapes(((10, 20), "bd"), ((10, 20, 20), "bdd"))
    True
    >>> check_shapes(((10, 20), "bd"), ((10, 30, 20), "bdd"), raise_on_errors=False)
    False
    """
    dims: dict[str, tuple[int, ...]] = dict()
    errors = []
    for actual_shape, shape in x:
        if isinstance(actual_shape, torch.Tensor):
            actual_shape = actual_shape.shape
        if len(actual_shape) != len(shape):
            errors.append(f"Invalid number of dimensions: {actual_shape} vs. {shape}")
            continue
        for dim, name in zip(actual_shape, shape):
            exp_dim = dims.get(name)
            if exp_dim is not None and exp_dim != dim:
                errors.append(f"{name}: {dim} vs. {exp_dim}")
            dims[name] = dim
    if raise_on_errors and errors:
        raise ValueError("Shape verification failed:\n" + "\n".join(errors))
    return len(errors) == 0


@functools.lru_cache(maxsize=1)
def get_expected_norm(
    p: int | float | str,
    d: int,
) -> float:
    r"""Compute the expected value of the L_p norm.

    .. math ::
        E[\|x\|_p] = d^{1/p} E[|x_1|^p]^{1/p}

    under the assumption that :math:`x_i \sim N(0, 1)`, i.e.

    .. math ::
        E[|x_1|^p] = 2^{p/2} \cdot \Gamma(\frac{p+1}{2} \cdot \pi^{-1/2}

    :param p:
        The parameter p of the norm.
    :param d:
        The dimension of the vector.

    :return:
        The expected value.

    :raises NotImplementedError: If infinity or negative infinity are given as p
    :raises TypeError: If an invalid type was given

    .. seealso ::
        https://math.stackexchange.com/questions/229033/lp-norm-of-multivariate-standard-normal-random-variable
        https://www.wolframalpha.com/input/?i=expected+value+of+%7Cx%7C%5Ep
    """
    if isinstance(p, str):
        p = float(p)
    if math.isinf(p) and p > 0:  # max norm
        # TODO: this only works for x ~ N(0, 1), but not for |x|
        raise NotImplementedError("Normalization for inf norm is not implemented")
        # cf. https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
        # mean = scipy.stats.norm.ppf(1 - 1/d)
        # scale = scipy.stats.norm.ppf(1 - 1/d * 1/math.e) - mean
        # return scipy.stats.gumbel_r.mean(loc=mean, scale=scale)
    elif math.isfinite(p):
        exp_abs_norm_p = math.pow(2, p / 2) * math.gamma((p + 1) / 2) / math.sqrt(math.pi)
        return math.pow(exp_abs_norm_p * d, 1 / p)
    else:
        raise TypeError(f"norm not implemented for {type(p)}: {p}")


class Bias(nn.Module):
    """A module wrapper for adding a bias."""

    def __init__(self, dim: int):
        """Initialize the module.

        :param dim: >0
            The dimension of the input.
        """
        super().__init__()
        self.bias = nn.Parameter(torch.empty(dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer's parameters."""
        nn.init.zeros_(self.bias)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """Add the learned bias to the input.

        :param x: shape: (n, d)
            The input.

        :return:
            x + b[None, :]
        """
        return x + self.bias.unsqueeze(dim=0)


def lp_norm(x: FloatTensor, p: float, dim: int | None, normalize: bool) -> FloatTensor:
    """Return the $L_p$ norm."""
    value = x.norm(p=p, dim=dim)
    if not normalize:
        return value
    return value / get_expected_norm(p=p, d=x.shape[-1])


def powersum_norm(x: FloatTensor, p: float, dim: int | None, normalize: bool) -> FloatTensor:
    """Return the power sum norm."""
    value = x.abs().pow(p).sum(dim=dim)
    if not normalize:
        return value
    dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
    return value / dim


def complex_normalize(x: torch.Tensor) -> torch.Tensor:
    r"""Normalize a vector of complex numbers such that each *element* is of unit-length.

    Let $x \in \mathbb{C}^d$ denote a complex vector. Then, the operation computes

    .. math::
        x_i' = \frac{x_i}{|x_i|}

    where $|x_i| = \sqrt{Re(x_i)^2 + Im(x_i)^2}$ is the
    `modulus of complex number <https://en.wikipedia.org/wiki/Absolute_value#Complex_numbers>`_

    :param x:
        A tensor formulating complex numbers

    :returns:
        An elementwise normalized vector.
    """
    if torch.is_complex(x):
        return x / x.abs().clamp_min(torch.finfo(x.dtype).eps)

    # note: this is a hack, and should be fixed up-stream by making NodePiece
    #  use proper complex embeddings for rotate interaction; however, we also have representations
    #  that perform message passing, and we would need to propagate the base representation's complexity through it
    warnings.warn(
        "Applying complex_normalize on non-complex input; if you see shape errors downstream this may be a possible "
        "root cause.",
        stacklevel=2,
    )
    (x_complex,) = ensure_complex(x)
    x_complex = complex_normalize(x_complex)
    x_real = torch.view_as_real(x_complex)
    return x_real.view(x.shape)


CONFIGURATION_FILE_FORMATS = {".json", ".yaml", ".yml"}


def load_configuration(path: str | pathlib.Path | os.PathLike) -> Mapping[str, Any]:
    """Load a configuration from a JSON or YAML file."""
    # ensure pathlib
    path = pathlib.Path(path)

    if path.suffix == ".json":
        with path.open() as file:
            return json.load(file)

    if path.suffix in {".yaml", ".yml"}:
        with path.open() as file:
            return yaml.safe_load(file)

    raise ValueError(f"Unknown configuration file format: {path.suffix}. Valid formats: {CONFIGURATION_FILE_FORMATS}")


def getattr_or_docdata(cls, key: str) -> str:
    """Get the attr or data inside docdata."""
    if hasattr(cls, key):
        return getattr(cls, key)
    getter_key = f"get_{key}"
    if hasattr(cls, getter_key):
        return getattr(cls, getter_key)()
    docdata = get_docdata(cls)
    if key in docdata:
        return docdata[key]
    raise KeyError


def triple_tensor_to_set(tensor: LongTensor) -> set[tuple[int, ...]]:
    """Convert a tensor of triples to a set of int-tuples."""
    return set(map(tuple, tensor.tolist()))


def is_triple_tensor_subset(a: LongTensor, b: LongTensor) -> bool:
    """Check whether one tensor of triples is a subset of another one."""
    return triple_tensor_to_set(a).issubset(triple_tensor_to_set(b))


def create_relation_to_entity_set_mapping(
    triples: Iterable[tuple[int, int, int]],
) -> tuple[Mapping[int, set[int]], Mapping[int, set[int]]]:
    """
    Create mappings from relation IDs to the set of their head / tail entities.

    :param triples:
        The triples.

    :return:
        A pair of dictionaries, each mapping relation IDs to entity ID sets.
    """
    tails = defaultdict(set)
    heads = defaultdict(set)
    for h, r, t in triples:
        heads[r].add(h)
        tails[r].add(t)
    return heads, tails


camel_to_snake_pattern = re.compile(r"(?<!^)(?=[A-Z])")


def camel_to_snake(name: str) -> str:
    """Convert camel-case to snake case."""
    # cf. https://stackoverflow.com/a/1176023
    return camel_to_snake_pattern.sub("_", name).lower()


def make_ones_like(prefix: Sequence) -> Sequence[int]:
    """Create a list of ones of same length as the input sequence."""
    return [1 for _ in prefix]


def logcumsumexp(a: np.ndarray) -> np.ndarray:
    """Compute ``log(cumsum(exp(a)))``.

    :param a: shape: s
        the array

    :return: shape s
        the log-cumsum-exp of the array

    .. seealso ::
        :func:`scipy.special.logsumexp` and :func:`torch.logcumsumexp`
    """
    a_max = np.amax(a)
    tmp = np.exp(a - a_max)
    s = np.cumsum(tmp)
    out = np.log(s)
    out += a_max
    return out


def find(x: X, parent: MutableMapping[X, X]) -> X:
    """Find step of union-find data structure with path compression."""
    # check validity
    if x not in parent:
        raise ValueError(f"Unknown element: {x}.")
    # path compression
    while parent[x] != x:
        x, parent[x] = parent[x], parent[parent[x]]
    return x


def get_connected_components(pairs: Iterable[tuple[X, X]]) -> Collection[Collection[X]]:
    """
    Calculate the connected components for a graph given as edge list.

    The implementation uses a `union-find <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>`_ data structure
    with path compression.

    :param pairs:
        the edge list, i.e., pairs of node ids.

    :return:
        a collection of connected components, i.e., a collection of disjoint collections of node ids.
    """
    parent: dict[X, X] = dict()
    for x, y in pairs:
        parent.setdefault(x, x)
        parent.setdefault(y, y)
        # get representatives
        x = find(x=x, parent=parent)
        y = find(x=y, parent=parent)
        # already merged
        if x == y:
            continue
        # make x the smaller one
        if y < x:  # type: ignore
            x, y = y, x
        # merge
        parent[y] = x
    # extract partitions
    result = defaultdict(list)
    for k, v in parent.items():
        result[v].append(k)
    return list(result.values())


PathType = Union[str, pathlib.Path, TextIO]


def normalize_path(
    path: PathType | None,
    *other: str | pathlib.Path,
    mkdir: bool = False,
    is_file: bool = False,
    default: PathType | None = None,
) -> pathlib.Path:
    """
    Normalize a path.

    :param path:
        the path in either of the valid forms.
    :param other:
        additional parts to join to the path
    :param mkdir:
        whether to ensure that the path refers to an existing directory by creating it if necessary
    :param is_file:
        whether the path is intended to be a file - only relevant for creating directories
    :param default:
        the default to use if path is None

    :raises TypeError:
        if `path` is of unsuitable type
    :raises ValueError:
        if `path` and `default` are both `None`

    :return:
        the absolute and resolved path
    """
    if path is None:
        if default is None:
            raise ValueError("If no default is provided, path cannot be None.")
        path = default
    if isinstance(path, TextIO):
        path = path.name
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not isinstance(path, pathlib.Path):
        raise TypeError(f"path is invalid type: {type(path)}")
    if other:
        path = path.joinpath(*other)
    # resolve path to make sure it is an absolute path
    path = path.expanduser().resolve()
    # ensure directory exists
    if mkdir:
        directory = path
        if is_file:
            directory = path.parent
        directory.mkdir(exist_ok=True, parents=True)
    return path


def ensure_complex(*xs: torch.Tensor) -> Iterable[torch.Tensor]:
    """
    Ensure that all tensors are of complex dtype.

    Reshape and convert if necessary.

    :param xs:
        the tensors

    :yields: complex tensors.
    """
    for x in xs:
        if x.is_complex():
            yield x
            continue
        warnings.warn(f"{x=} is not complex, but will be viewed as such", stacklevel=2)
        if x.shape[-1] != 2:
            x = x.view(*x.shape[:-1], -1, 2)
        yield torch.view_as_complex(x)


def _weisfeiler_lehman_iteration(
    adj: torch.Tensor,
    colors: LongTensor,
    dense_dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """
    Perform a single Weisfeiler-Lehman iteration.

    :param adj: shape: `(n, n)`
        the adjacency matrix
    :param colors: shape: `(n,)`
        the node colors (as integers)
    :param dense_dtype:
        a datatype for storing integers of sufficient capacity to store `n`

    :return: shape: `(n,)`
        the new node colors
    """
    num_nodes = colors.shape[0]
    # message passing: collect colors of neighbors
    # dense colors: shape: (n, c)
    # adj:          shape: (n, n)
    color_sparse = torch.sparse_coo_tensor(
        indices=torch.stack([torch.arange(num_nodes, device=colors.device), colors], dim=0),
        # values need to be float, since torch.sparse.mm does not support integer dtypes
        values=torch.ones_like(colors, dtype=torch.get_default_dtype()),
        # size: will be correctly inferred
    )
    color_dense = torch.sparse.mm(adj, color_sparse).to(dtype=dense_dtype).to_dense()

    # concat with old colors
    colors = torch.cat([colors.unsqueeze(dim=-1), color_dense], dim=-1)

    # hash
    return colors.unique(dim=0, return_inverse=True)[1]


def _weisfeiler_lehman_iteration_approx(
    adj: torch.Tensor,
    colors: LongTensor,
    dim: int = 32,
    decimals: int = 6,
) -> torch.Tensor:
    """
    Perform an approximate  single Weisfeiler-Lehman iteration.

    It utilizes the trick from https://arxiv.org/abs/2001.09621 of replacing node indicator functions by
    randomly drawn functions of fixed and low dimensionality.

    :param adj: shape: `(n, n)`
        the adjacency matrix
    :param colors: shape: `(n,)`
        the node colors (as integers)
    :param dim:
        the dimensionality of the random node indicator functions
    :param decimals:
        the number of decimals to round to

    :return: shape: `(n,)`
        the new node colors
    """
    num_colors = colors.max() + 1

    # create random indicator functions of low dimensionality
    rand = torch.rand(num_colors, dim, device=colors.device)
    x = rand[colors]

    # collect neighbors' colors
    x = torch.sparse.mm(adj, x)

    # round to avoid numerical effects
    x = torch.round(x, decimals=decimals)

    # hash first
    new_colors = x.unique(return_inverse=True, dim=0)[1]

    # concat with old colors
    colors = torch.stack([colors, new_colors], dim=-1)

    # re-hash
    return colors.unique(return_inverse=True, dim=0)[1]


def iter_weisfeiler_lehman(
    edge_index: LongTensor,
    max_iter: int = 2,
    num_nodes: int | None = None,
    approximate: bool = False,
) -> Iterable[torch.Tensor]:
    """
    Iterate Weisfeiler-Lehman colors.

    The Weisfeiler-Lehman algorithm starts from a uniformly node-colored graph, and iteratively counts the colors in
    each nodes' neighborhood, and hashes these multi-sets into a new set of colors.

    .. note::
        more precisely, this implementation calculates the results of the 1-Weisfeiler-Lehman algorithm. There is a
        hierarchy of k-WL tests, which color k-tuples of nodes instead of single nodes

    .. note::
        Graph Neural Networks are closely related to the 1-WL test, cf. e.g., https://arxiv.org/abs/1810.02244

    .. note::
        colorings based on the WL test have been used to define graph kernels, cf., e.g.,
        https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf

    .. note::
        the implementation creates intermediate dense tensors of shape `(num_nodes, num_colors)`

    .. seealso::
        https://towardsdatascience.com/expressive-power-of-graph-neural-networks-and-the-weisefeiler-lehman-test-b883db3c7c49

    :param edge_index: shape: `(2, m)`
        the edge list
    :param max_iter:
        the maximum number of iterations
    :param num_nodes:
        the number of nodes. If None, will be inferred from the edge index.
    :param approximate:
        whether to use an approximate, but more memory-efficient implementation.

    :raises ValueError:
        if the number of nodes exceeds `torch.long` (this cannot happen in practice, as the edge index tensor
        construction would already fail earlier)

    :yields: the colors for each Weisfeiler-Lehman iteration
    """
    # only keep connectivity, but remove multiplicity
    edge_index = edge_index.unique(dim=1)

    num_nodes = num_nodes or edge_index.max().item() + 1
    colors = edge_index.new_zeros(size=(num_nodes,), dtype=torch.long)
    # note: in theory, we could return this uniform coloring as the first coloring; however, for featurization,
    #       this is rather useless

    # initial: degree
    # note: we calculate this separately, since we can use a more efficient implementation for the first step
    unique, counts = edge_index.unique(return_counts=True)
    colors[unique] = counts

    # hash
    colors = colors.unique(return_inverse=True)[1]
    num_colors = colors.max() + 1
    yield colors

    # determine small integer type for dense count array
    for idtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        if torch.iinfo(idtype).max >= num_nodes:
            dense_dtype = idtype
            logger.debug(f"Selected dense dtype: {dense_dtype}")
            break
    else:
        raise ValueError(f"{num_nodes} too large")

    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones(size=edge_index[0].shape),
        device=edge_index.device,
        size=(num_nodes, num_nodes),
    )
    for i in range(2, max_iter + 1):
        if approximate:
            colors = _weisfeiler_lehman_iteration_approx(adj=adj, colors=colors)
        else:
            colors = _weisfeiler_lehman_iteration(adj=adj, colors=colors, dense_dtype=dense_dtype)
        yield colors

        # convergence check
        new_num_colors = colors.max() + 1

        # each node has a unique color
        if new_num_colors >= (num_nodes - 1):
            logger.debug(f"Weisfeiler-Lehman terminated with unique colors for each node after {i} iterations")
            break

        # the number of colors did not improve in the last iteration
        if num_colors >= new_num_colors:
            logger.debug(f"Weisfeiler-Lehman could not further refine coloring in iteration {i}")
            break

        num_colors = new_num_colors
    else:
        logger.debug(f"Weisfeiler-Lehman did not converge after {max_iter} iterations.")


def get_edge_index(
    *,
    # cannot use Optional[pykeen.triples.CoreTriplesFactory] due to cyclic imports
    triples_factory: Any | None = None,
    mapped_triples: MappedTriples | None = None,
    edge_index: LongTensor | None = None,
) -> LongTensor:
    """
    Get the edge index from a number of different sources.

    :param triples_factory:
        the triples factory
    :param mapped_triples: shape: `(m, 3)`
        ID-based triples
    :param edge_index: shape: `(2, m)`
        the edge index

    :raises ValueError:
        if none of the source was different from `None`

    :return: shape: `(2, m)`
        the edge index
    """
    if triples_factory is not None:
        if mapped_triples is not None:
            logger.warning("Ignoring mapped_triples, since triples_factory is present.")
        mapped_triples = triples_factory.mapped_triples
    if mapped_triples is not None:
        if edge_index is not None:
            logger.warning("Ignoring edge_index, since mapped_triples is present.")
        edge_index = mapped_triples[:, 0::2].t()
    if edge_index is None:
        raise ValueError("At least one of the parameters must be different to None.")
    return edge_index


def prepare_filter_triples(
    mapped_triples: MappedTriples,
    additional_filter_triples: None | MappedTriples | list[MappedTriples] = None,
    warn: bool = True,
) -> MappedTriples:
    """Prepare the filter triples from the evaluation triples, and additional filter triples."""
    if torch.is_tensor(additional_filter_triples):
        additional_filter_triples = [additional_filter_triples]
    if additional_filter_triples is not None:
        return torch.cat([*additional_filter_triples, mapped_triples], dim=0).unique(dim=0)
    if warn:
        logger.warning(
            dedent(
                """\
                The filtered setting was enabled, but there were no `additional_filter_triples`
                given. This means you probably forgot to pass (at least) the training triples. Try:

                    additional_filter_triples=[dataset.training.mapped_triples]

                Or if you want to use the Bordes et al. (2013) approach to filtering, do:

                    additional_filter_triples=[
                        dataset.training.mapped_triples,
                        dataset.validation.mapped_triples,
                    ]
                """
            )
        )
    return mapped_triples


def nested_get(d: Mapping[str, Any], *key: str, default=None) -> Any:
    """
    Get from a nested dictionary.

    :param d:
        the (nested) dictionary
    :param key:
        a sequence of keys
    :param default:
        the default value

    :return:
        the value or default
    """
    for k in key:
        if k not in d:
            return default
        d = d[k]
    return d


def rate_limited(xs: Iterable[X], min_avg_time: float = 1.0) -> Iterable[X]:
    """Iterate over iterable with rate limit.

    :param xs:
        the iterable
    :param min_avg_time:
        the minimum average time per element

    :yields: elements of the iterable
    """
    start = time.perf_counter()
    for i, x in enumerate(xs):
        duration = time.perf_counter() - start
        under = min_avg_time * i - duration
        under = max(0, under)
        if under:
            logger.debug(f"Applying rate limit; sleeping for {under} seconds")
            time.sleep(under)
        yield x


class ExtraReprMixin:
    """
    A mixin for modules with hierarchical `extra_repr`.

    It takes up the :meth:`torch.nn.Module.extra_repr` idea, and additionally provides a simple
    composable way to generate the components of :meth:`extra_repr` via :meth:`iter_extra_repr`.

    If combined with `torch.nn.Module`, make sure to put :class:`ExtraReprMixin` *behind*
    :class:`torch.nn.Module` to prefer the latter's :func:`__repr__` implementation.
    """

    def iter_extra_repr(self) -> Iterable[str]:
        """
        Iterate over the components of the :meth:`extra_repr`.

        This method is typically overridden. A common pattern would be

        .. code-block:: python

            def iter_extra_repr(self) -> Iterable[str]:
                yield from super().iter_extra_repr()
                yield "<key1>=<value1>"
                yield "<key2>=<value2>"

        :return:
            an iterable over individual components of the :meth:`extra_repr`
        """
        return []

    def extra_repr(self) -> str:
        """
        Generate the extra repr, cf. :meth`torch.nn.Module.extra_repr`.

        :return:
            the extra part of the :func:`repr`
        """
        return ", ".join(self.iter_extra_repr())

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}({self.extra_repr()})"


try:
    from opt_einsum import contract

    einsum = functools.partial(contract, backend="torch")
    logger.info("Using opt_einsum")
except ImportError:
    einsum = torch.einsum


def isin_many_dim(elements: torch.Tensor, test_elements: torch.Tensor, dim: int = 0) -> BoolTensor:
    """Return whether elements are contained in test elements."""
    inverse, counts = torch.cat([elements, test_elements], dim=dim).unique(
        return_counts=True, return_inverse=True, dim=dim
    )[1:]
    return counts[inverse[: elements.shape[dim]]] > 1


def determine_maximum_batch_size(batch_size: int | None, device: torch.device, maximum_batch_size: int) -> int:
    """Normalize choice of maximum batch size.

    On non-CUDA devices, excessive batch sizes might cause out of memory errors from which the program cannot recover.

    :param batch_size:
        The chosen (maximum) batch size, or `None` when the largest possible one should be used.
    :param device:
        The device on which the evaluation runs.
    :param maximum_batch_size:
        The actual maximum batch size, e.g., the size of the evaluation set.

    :return:
        A maximum batch size.
    """
    if batch_size is None:
        if device.type == "cuda":
            batch_size = maximum_batch_size
        else:
            batch_size = 32
            logger.warning(
                f"Using automatic batch size on {device.type=} can cause unexplained out-of-memory crashes. "
                f"Therefore, we use a conservative small {batch_size=:_}. "
                f"Performance may be improved by explicitly specifying a larger batch size."
            )
        logger.debug(f"Automatically set maximum batch size to {batch_size=:_}")
    return batch_size


def add_cudnn_error_hint(func: Callable[P, X]) -> Callable[P, X]:
    """
    Decorate a function to add explanations for CUDNN errors.

    :param func:
        the function to decorate

    :return:
        a decorated function
    """

    # docstr-coverage: excused `wrapped`
    @functools.wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> X:
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if not is_cudnn_error(e):
                raise e
            raise RuntimeError(
                "\nThis code crash might have been caused by a CUDA bug, see "
                "https://github.com/allenai/allennlp/issues/2888, "
                "which causes the code to crash during evaluation mode.\n"
                "To avoid this error, the batch size has to be reduced.",
            ) from e

    return wrapped


def split_workload(n: int) -> range:
    """Split workload for multi-processing."""
    # cf. https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # single-process data loading, return the full iterator
        workload = range(n)
    else:
        num_workers = worker_info.num_workers
        worker_id = worker_info.id  # 1-based
        start = math.ceil(n / num_workers * worker_id)
        stop = math.ceil(n / num_workers * (worker_id + 1))
        workload = range(start, stop)
    return workload


def batched_dot(a: FloatTensor, b: FloatTensor) -> FloatTensor:
    """Compute "element-wise" dot-product between batched vectors."""
    return (a * b).sum(dim=-1)


def _batched_dot_matmul(a: FloatTensor, b: FloatTensor) -> FloatTensor:
    """Compute "element-wise" dot-product between batched vectors."""
    return (a.unsqueeze(dim=-2) @ b.unsqueeze(dim=-1)).view(a.shape[:-1])


def _batched_dot_einsum(a: FloatTensor, b: FloatTensor) -> FloatTensor:
    return einsum("...i,...i->...", a, b)


def circular_correlation(a: FloatTensor, b: FloatTensor) -> FloatTensor:
    """
    Compute the circular correlation between to vectors.

    .. note ::
        The implementation uses FFT.

    :param a: shape: s_1
        The tensor with the first vectors.
    :param b:
        The tensor with the second vectors.

    :return:
        The circular correlation between the vectors.
    """
    # Circular correlation of entity embeddings
    a_fft = torch.fft.rfft(a, dim=-1)
    b_fft = torch.fft.rfft(b, dim=-1)
    # complex conjugate
    a_fft = torch.conj(a_fft)
    # Hadamard product in frequency domain
    p_fft = a_fft * b_fft
    # inverse real FFT
    return torch.fft.irfft(p_fft, n=a.shape[-1], dim=-1)
