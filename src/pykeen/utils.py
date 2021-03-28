# -*- coding: utf-8 -*-

"""Utilities for PyKEEN."""

import ftplib
import functools
import inspect
import itertools as itt
import json
import logging
import math
import operator
import random
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import (
    Any, Callable, Collection, Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.modules.batchnorm
from class_resolver import Resolver, normalize_string
from torch import nn
from torch.nn import functional

from .constants import PYKEEN_BENCHMARKS
from .typing import DeviceHint, MappedTriples, TorchRandomHint
from .version import get_git_hash

__all__ = [
    'compose',
    'clamp_norm',
    'compact_mapping',
    'ensure_torch_random_state',
    'format_relative_comparison',
    'invert_mapping',
    'is_cuda_oom_error',
    'random_non_negative_int',
    'resolve_device',
    'split_complex',
    'split_list_in_batches_iter',
    'torch_is_in_1d',
    'normalize_string',
    'get_until_first_blank',
    'flatten_dictionary',
    'set_random_seed',
    'NoRandomSeedNecessary',
    'Result',
    'fix_dataclass_init_docs',
    'get_benchmark',
    'extended_einsum',
    'strip_dim',
    'upgrade_to_sequence',
    'ensure_tuple',
    'unpack_singletons',
    'extend_batch',
    'check_shapes',
    'all_in_bounds',
    'is_cudnn_error',
    'view_complex',
    'combine_complex',
    'get_model_io',
    'get_json_bytes_io',
    'get_df_io',
    'ensure_ftp_directory',
    'broadcast_cat',
    'get_batchnorm_modules',
    'calculate_broadcasted_elementwise_result_shape',
    'estimate_cost_of_sequence',
    'get_optimal_sequence',
    'tensor_sum',
    'tensor_product',
    'negative_norm_of_sum',
    'negative_norm',
    'project_entity',
    'CANONICAL_DIMENSIONS',
    'convert_to_canonical_shape',
    'get_expected_norm',
    'Bias',
    'activation_resolver',
    'complex_normalize',
    'lp_norm',
    'powersum_norm',
]

logger = logging.getLogger(__name__)

#: An error that occurs because the input in CUDA is too big. See ConvE for an example.
_CUDNN_ERROR = 'cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.'

_CUDA_OOM_ERROR = 'CUDA out of memory.'

_CUDA_NONZERO_ERROR = "nonzero is not supported for tensors with more than INT_MAX elements"


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
    dictionary: Mapping[str, Any],
    prefix: Optional[str] = None,
    sep: str = '.',
) -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    real_prefix = tuple() if prefix is None else (prefix,)
    partial_result = _flatten_dictionary(dictionary=dictionary, prefix=real_prefix)
    return {sep.join(map(str, k)): v for k, v in partial_result.items()}


def _flatten_dictionary(
    dictionary: Mapping[str, Any],
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
        A vector with $|x| <= maxnorm$.
    """
    norm = x.norm(p=p, dim=dim, keepdim=True)
    mask = (norm < maxnorm).type_as(x)
    return mask * x + (1 - mask) * (x / norm.clamp_min(eps) * maxnorm)


class compose(Generic[X]):  # noqa:N801
    """A class representing the composition of several functions."""

    def __init__(self, *operations: Callable[[X], X]):
        """Initialize the composition with a sequence of operations.

        :param operations: unary operations that will be applied in succession
        """
        self.operations = operations

    def __call__(self, x: X) -> X:
        """Apply the operations in order to the given tensor."""
        for operation in self.operations:
            x = operation(x)
        return x


def set_random_seed(seed: int) -> Tuple[None, torch.Generator, None]:
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

    :returns: If all values are within the given bounds
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


def is_nonzero_larger_than_maxint_error(runtime_error: RuntimeError) -> bool:
    """Check if the runtime error was caused by applying nonzero to a GPU tensor with more than ``MAX_INT`` elements.

    :param runtime_error: The exception to check
    :returns: if the exception is a runtime error caused by func:`torch.nonzero` being applied to a GPU tensor with
        more than ``MAX_INT`` elements

    .. seealso:: https://github.com/pytorch/pytorch/issues/51871
    """
    return _CUDA_NONZERO_ERROR in runtime_error.args[0]


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


def view_complex(x: torch.FloatTensor) -> torch.Tensor:
    """Convert a PyKEEN complex tensor representation into a torch one."""
    real, imag = split_complex(x=x)
    return torch.complex(real=real, imag=imag)


def combine_complex(
    x_re: torch.FloatTensor,
    x_im: torch.FloatTensor,
) -> torch.FloatTensor:
    """Combine a complex tensor from real and imaginary part."""
    return torch.cat([x_re, x_im], dim=-1)


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

    :raises ValueError: if the mapping is not bijective
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
    Return a boolean mask with ``Q[i]`` in T.

    The method guarantees memory complexity of ``max(size(Q), size(T))`` and is thus, memory-wise, superior to naive
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


def broadcast_cat(
    tensors: Sequence[torch.FloatTensor],
    dim: int,
) -> torch.FloatTensor:
    """Concatenate tensors with broadcasting support.

    :param tensors:
        The tensors. Each of the tensors is require to have the same number of dimensions.
        For each dimension not equal to dim, the extent has to match the other tensors', or be one.
        If it is one, the tensor is repeated to match the extent of the othe tensors.
    :param dim:
        The concat dimension.

    :return: A concatenated, broadcasted tensor.

    :raises ValueError: if the x and y dimensions are not the same
    :raises ValueError: if broadcasting is not possible
    """
    # input validation
    if len(tensors) == 0:
        raise ValueError("Must pass at least one tensor.")
    if len({x.ndimension() for x in tensors}) != 1:
        raise ValueError(
            f"The number of dimensions has to be the same for all tensors, but is {set(t.shape for t in tensors)}",
        )

    # base case
    if len(tensors) == 1:
        return tensors[0]

    # normalize dim
    if dim < 0:
        dim = tensors[0].ndimension() + dim

    # calculate repeats for each tensor
    repeats = [
        [1 for _ in t.shape]
        for t in tensors
    ]
    for i, dims in enumerate(zip(*(t.shape for t in tensors))):
        # dimensions along concatenation axis do not need to match
        if i == dim:
            continue

        # get desired extent along dimension
        d_max = max(dims)
        if not {1, d_max}.issuperset(dims):
            raise ValueError(f"Tensors have invalid shape along {i} dimension: {set(dims)}")

        for j, td in enumerate(dims):
            if td != d_max:
                repeats[j][i] = d_max

    # repeat tensors along axes if necessary
    tensors = [
        t.repeat(*r)
        for t, r in zip(tensors, repeats)
    ]

    # concatenate
    return torch.cat(tensors, dim=dim)


def get_batchnorm_modules(module: torch.nn.Module) -> List[torch.nn.Module]:
    """Return all submodules which are batch normalization layers."""
    return [
        submodule
        for submodule in module.modules()
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm)
    ]


def calculate_broadcasted_elementwise_result_shape(
    first: Tuple[int, ...],
    second: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Determine the return shape of a broadcasted elementwise operation."""
    return tuple(max(a, b) for a, b in zip(first, second))


def estimate_cost_of_sequence(
    shape: Tuple[int, ...],
    *other_shapes: Tuple[int, ...],
) -> int:
    """Cost of a sequence of broadcasted element-wise operations of tensors, given their shapes."""
    return sum(map(
        np.prod,
        itt.islice(
            itt.accumulate(
                (shape,) + other_shapes,
                calculate_broadcasted_elementwise_result_shape,
            ),
            1,
            None,
        ),
    ))


@functools.lru_cache(maxsize=32)
def _get_optimal_sequence(
    *sorted_shapes: Tuple[int, ...],
) -> Tuple[int, Tuple[int, ...]]:
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
def get_optimal_sequence(*shapes: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
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
    tensors: Tuple[torch.FloatTensor, ...],
) -> Tuple[torch.FloatTensor, ...]:
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
    if len(set(s[0] for s in shapes)) < 2:
        # heuristic
        return tensors
    order = get_optimal_sequence(*shapes)[1]
    return tuple(tensors[i] for i in order)


def tensor_sum(*tensors: torch.FloatTensor) -> torch.FloatTensor:
    """Compute element-wise sum of tensors in broadcastable shape."""
    return sum(_reorder(tensors=tensors))


def tensor_product(*tensors: torch.FloatTensor) -> torch.FloatTensor:
    """Compute element-wise product of tensors in broadcastable shape."""
    head, *rest = _reorder(tensors=tensors)
    return functools.reduce(operator.mul, rest, head)


def negative_norm_of_sum(
    *x: torch.FloatTensor,
    p: Union[str, int, float] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """Evaluate negative norm of a sum of vectors on already broadcasted representations.

    :param x: shape: (batch_size, num_heads, num_relations, num_tails, dim)
        The representations.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return $|x-y|_p^p$, cf. https://github.com/pytorch/pytorch/issues/28119

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm(tensor_sum(*x), p=p, power_norm=power_norm)


def negative_norm(
    x: torch.FloatTensor,
    p: Union[str, int, float] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """Evaluate negative norm of a vector.

    :param x: shape: (batch_size, num_heads, num_relations, num_tails, dim)
        The vectors.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return $|x-y|_p^p$, cf. https://github.com/pytorch/pytorch/issues/28119

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    if power_norm:
        assert not isinstance(p, str)
        return -(x.abs() ** p).sum(dim=-1)

    if torch.is_complex(x):
        assert not isinstance(p, str)
        # workaround for complex numbers: manually compute norm
        return -(x.abs() ** p).sum(dim=-1) ** (1 / p)

    return -x.norm(p=p, dim=-1)


def extended_einsum(
    eq: str,
    *tensors,
) -> torch.FloatTensor:
    """Drop dimensions of size 1 to allow broadcasting."""
    # TODO: check if einsum is still very slow.
    lhs, rhs = eq.split("->")
    mod_ops, mod_t = [], []
    for op, t in zip(lhs.split(","), tensors):
        mod_op = ""
        if len(op) != len(t.shape):
            raise ValueError(f'Shapes not equal: op={op} and t.shape={t.shape}')
        # TODO: t_shape = list(t.shape); del t_shape[i]; t.view(*shape) -> only one reshape operation
        for i, c in reversed(list(enumerate(op))):
            if t.shape[i] == 1:
                t = t.squeeze(dim=i)
            else:
                mod_op = c + mod_op
        mod_ops.append(mod_op)
        mod_t.append(t)
    m_lhs = ",".join(mod_ops)
    r_keep_dims = set("".join(mod_ops))
    m_rhs = "".join(c for c in rhs if c in r_keep_dims)
    m_eq = f"{m_lhs}->{m_rhs}"
    mod_r = torch.einsum(m_eq, *mod_t)
    # unsqueeze
    for i, c in enumerate(rhs):
        if c not in r_keep_dims:
            mod_r = mod_r.unsqueeze(dim=i)
    return mod_r


def project_entity(
    e: torch.FloatTensor,
    e_p: torch.FloatTensor,
    r_p: torch.FloatTensor,
) -> torch.FloatTensor:
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


CANONICAL_DIMENSIONS = dict(h=1, r=2, t=3)


def _normalize_dim(dim: Union[int, str]) -> int:
    """Normalize the dimension selection."""
    if isinstance(dim, int):
        return dim
    return CANONICAL_DIMENSIONS[dim.lower()[0]]


def convert_to_canonical_shape(
    x: torch.FloatTensor,
    dim: Union[int, str],
    num: Optional[int] = None,
    batch_size: int = 1,
    suffix_shape: Union[int, Sequence[int]] = -1,
) -> torch.FloatTensor:
    """Convert a tensor to canonical shape.

    :param x:
        The tensor in compatible shape.
    :param dim:
        The "num" dimension.
    :param batch_size:
        The batch size.
    :param num:
        The number.
    :param suffix_shape:
        The suffix shape.

    :return: shape: (batch_size, num_heads, num_relations, num_tails, ``*``)
        A tensor in canonical shape.
    """
    if num is None:
        num = x.shape[0]
    suffix_shape = upgrade_to_sequence(suffix_shape)
    shape = [batch_size, 1, 1, 1]
    dim = _normalize_dim(dim=dim)
    shape[dim] = num
    return x.view(*shape, *suffix_shape)


def strip_dim(*tensors: torch.FloatTensor, n: int = 4) -> Sequence[torch.FloatTensor]:
    """Strip the first dimensions.

    :param tensors: The tensors whose first ``n`` dimensions should be independently stripped
    :param n: The number of initial dimensions to strip
    :return: A tuple of the reduced tensors
    """
    return tuple(tensor.view(tensor.shape[n:]) for tensor in tensors)


def upgrade_to_sequence(x: Union[X, Sequence[X]]) -> Sequence[X]:
    """Ensure that the input is a sequence.

    :param x: A literal or sequence of literals
    :return: If a literal was given, a one element tuple with it in it. Otherwise, return the given value.

    >>> upgrade_to_sequence(1)
    (1,)
    >>> upgrade_to_sequence((1, 2, 3))
    (1, 2, 3)
    """
    return x if isinstance(x, Sequence) else (x,)


def ensure_tuple(*x: Union[X, Sequence[X]]) -> Sequence[Sequence[X]]:
    """Ensure that all elements in the sequence are upgraded to sequences.

    :param x: A sequence of sequences or literals
    :return: An upgraded sequence of sequences

    >>> ensure_tuple(1, (1,), (1, 2))
    ((1,), (1,), (1, 2))
    """
    return tuple(upgrade_to_sequence(xx) for xx in x)


def unpack_singletons(*xs: Tuple[X]) -> Sequence[Union[X, Tuple[X]]]:
    """Unpack sequences of length one.

    :param xs: A sequence of tuples of length 1 or more
    :return: An unpacked sequence of sequences

    >>> unpack_singletons((1,), (1, 2), (1, 2, 3))
    (1, (1, 2), (1, 2, 3))
    """
    return tuple(
        x[0] if len(x) == 1 else x
        for x in xs
    )


def _can_slice(fn) -> bool:
    """Check if a model's score_X function can slice."""
    return 'slice_size' in inspect.getfullargspec(fn).args


def extend_batch(
    batch: MappedTriples,
    all_ids: List[int],
    dim: int,
) -> MappedTriples:
    """Extend batch for 1-to-all scoring by explicit enumeration.

    :param batch: shape: (batch_size, 2)
        The batch.
    :param all_ids: len: num_choices
        The IDs to enumerate.
    :param dim: in {0,1,2}
        The column along which to insert the enumerated IDs.

    :return: shape: (batch_size * num_choices, 3)
        A large batch, where every pair from the original batch is combined with every ID.
    """
    # Extend the batch to the number of IDs such that each pair can be combined with all possible IDs
    extended_batch = batch.repeat_interleave(repeats=len(all_ids), dim=0)

    # Create a tensor of all IDs
    ids = torch.tensor(all_ids, dtype=torch.long, device=batch.device)

    # Extend all IDs to the number of pairs such that each ID can be combined with every pair
    extended_ids = ids.repeat(batch.shape[0])

    # Fuse the extended pairs with all IDs to a new (h, r, t) triple tensor.
    columns = [extended_batch[:, i] for i in (0, 1)]
    columns.insert(dim, extended_ids)
    hrt_batch = torch.stack(columns, dim=-1)

    return hrt_batch


def check_shapes(
    *x: Tuple[Union[torch.Tensor, Tuple[int, ...]], str],
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
    dims: Dict[str, Tuple[int, ...]] = dict()
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
        raise ValueError("Shape verification failed:\n" + '\n'.join(errors))
    return len(errors) == 0


@functools.lru_cache(maxsize=1)
def get_expected_norm(
    p: Union[int, float, str],
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


activation_resolver = Resolver(
    classes=(
        nn.LeakyReLU,
        nn.PReLU,
        nn.ReLU,
        nn.Softplus,
        nn.Sigmoid,
        nn.Tanh,
    ),
    base=nn.Module,  # type: ignore
    default=nn.ReLU,
)


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

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Add the learned bias to the input.

        :param x: shape: (n, d)
            The input.

        :return:
            x + b[None, :]
        """
        return x + self.bias.unsqueeze(dim=0)


def lp_norm(x: torch.FloatTensor, p: float, dim: Optional[int], normalize: bool) -> torch.FloatTensor:
    """Return the $L_p$ norm."""
    value = x.norm(p=p, dim=dim)
    if not normalize:
        return value
    return value / get_expected_norm(p=p, d=x.shape[-1])


def powersum_norm(x: torch.FloatTensor, p: float, dim: Optional[int], normalize: bool) -> torch.FloatTensor:
    """Return the power sum norm."""
    value = x.abs().pow(p).sum(dim=dim)
    if not normalize:
        return value
    dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
    return value / dim


def complex_normalize(x: torch.Tensor) -> torch.Tensor:
    r"""Normalize a vector of complex numbers such that each element is of unit-length.

    :param x: A tensor formulating complex numbers
    :returns: A normalized version accoring to the following definition.

    The `modulus of complex number <https://en.wikipedia.org/wiki/Absolute_value#Complex_numbers>`_ is given as:

    .. math::

        |a + ib| = \sqrt{a^2 + b^2}

    $l_2$ norm of complex vector $x \in \mathbb{C}^d$:

    .. math::
        \|x\|^2 = \sum_{i=1}^d |x_i|^2
                 = \sum_{i=1}^d \left(\operatorname{Re}(x_i)^2 + \operatorname{Im}(x_i)^2\right)
                 = \left(\sum_{i=1}^d \operatorname{Re}(x_i)^2) + (\sum_{i=1}^d \operatorname{Im}(x_i)^2\right)
                 = \|\operatorname{Re}(x)\|^2 + \|\operatorname{Im}(x)\|^2
                 = \| [\operatorname{Re}(x); \operatorname{Im}(x)] \|^2
    """
    y = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    y = functional.normalize(y, p=2, dim=-1)
    return y.view(*x.shape)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
