# -*- coding: utf-8 -*-

"""Utilities for PyKEEN."""

import ftplib
import functools
import itertools
import json
import logging
import operator
import random
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.nn import functional

from .typing import DeviceHint

# get_subclasses, project_entity, set_random_seed, strip_dim, view_complex
__all__ = [
    'broadcast_cat',
    'compose',
    'check_shapes',
    'clamp_norm',
    'combine_complex',
    'compact_mapping',
    'complex_normalize',
    'fix_dataclass_init_docs',
    'flatten_dictionary',
    'get_cls',
    'get_subclasses',
    'get_until_first_blank',
    'invert_mapping',
    'is_cudnn_error',
    'is_cuda_oom_error',
    'project_entity',
    'negative_norm_of_sum',
    'normalize_string',
    'normalized_lookup',
    'random_non_negative_int',
    'resolve_device',
    'set_random_seed',
    'split_complex',
    'split_list_in_batches_iter',
    'upgrade_to_sequence',
    'view_complex',
    'NoRandomSeedNecessary',
    'Result',
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


def compose(
    *op_: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compose functions working on a single tensor."""

    def chained_op(x: torch.Tensor):
        for op in op_:
            x = op(x)
        return x

    return chained_op


def set_random_seed(seed: int) -> Tuple[None, torch._C.Generator, None]:
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


def invert_mapping(mapping: Mapping[str, int]) -> Mapping[int, str]:
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


def check_shapes(
    *x: Tuple[Union[torch.Tensor, Tuple[int, ...]], str],
    raise_on_errors: bool = True,
) -> bool:
    """
    Verify that a sequence of tensors are of matching shapes.

    :param x:
        A tuple (tensor, shape), where tensor is a tensor, and shape is a string, where each character corresponds to
        a (named) dimension. If the shapes of different tensors share a character, the corresponding dimensions are
        expected to be of equal size.
    :param raise_on_errors:
        Whether to raise an exception in case of a mismatch.

    :return:
        Whether the shapes matched.

    :raises ValueError:
        If the shapes mismatch and raise_on_error is True.
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


def broadcast_cat(
    x: torch.FloatTensor,
    y: torch.FloatTensor,
    dim: int,
) -> torch.FloatTensor:
    """
    Concatenate with broadcasting.

    :param x:
        The first tensor.
    :param y:
        The second tensor.
    :param dim:
        The concat dimension.

    :return:
    """
    if x.ndimension() != y.ndimension():
        raise ValueError
    if dim < 0:
        dim = x.ndimension() + dim
    x_rep, y_rep = [], []
    for d, (xd, yd) in enumerate(zip(x.shape, y.shape)):
        xr = yr = 1
        if d != dim and xd != yd:
            if xd == 1:
                xr = yd
            elif yd == 1:
                yr = xd
            else:
                raise ValueError
        x_rep.append(xr)
        y_rep.append(yr)
    return torch.cat([x.repeat(*x_rep), y.repeat(*y_rep)], dim=dim)


def get_subclasses(cls: Type[X]) -> Iterable[Type[X]]:
    """
    Get all subclasses.

    Credit to: https://stackoverflow.com/a/33607093.

    """
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass


def complex_normalize(x: torch.Tensor) -> torch.Tensor:
    r"""Normalize the length of relation vectors, if the forward constraint has not been applied yet.

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
    y = x.data.view(x.shape[0], -1, 2)
    y = functional.normalize(y, p=2, dim=-1)
    x.data = y.view(*x.shape)
    return x


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
        numpy.prod,
        itertools.islice(
            itertools.accumulate(
                other_shapes,
                calculate_broadcasted_elementwise_result_shape,
                initial=shape,
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
        for p in itertools.permutations(list(range(len(sorted_shapes))))
    )


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


def _multi_combine(
    tensors: Tuple[torch.FloatTensor, ...],
    op: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
) -> torch.FloatTensor:
    """Broadcasted element-wise combination of tensors.

    The optimal execution plan gets cached so that the optimization is only performed once for a fixed set of shapes.

    :param tensors:
        The tensors, in broadcastable shape.
    :param op:
        The elementwise operator.

    :return:
        The elementwise combination evaluated in optimal processing order.
    """
    # determine optimal processing order
    _, order = get_optimal_sequence(*(t.shape for t in tensors))
    head, *rest = [tensors[i] for i in order]
    return functools.reduce(op, rest, head)


def tensor_sum(*x: torch.FloatTensor) -> torch.FloatTensor:
    """Compute elementwise sum of tensors in brodcastable shape."""
    return _multi_combine(tensors=x, op=operator.add)


def tensor_product(*x: torch.FloatTensor) -> torch.FloatTensor:
    """Compute elementwise product of tensors in broadcastable shape."""
    return _multi_combine(tensors=x, op=operator.mul)


def negative_norm_of_sum(
    *x: torch.FloatTensor,
    p: Union[str, int] = 2,
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
    d: torch.FloatTensor = tensor_sum(*x)
    if power_norm:
        assert not isinstance(p, str)
        return -(d.abs() ** p).sum(dim=-1)

    if torch.is_complex(d):
        assert not isinstance(p, str)
        # workaround for complex numbers: manually compute norm
        return -(d.abs() ** p).sum(dim=-1) ** (1 / p)

    return -d.norm(p=p, dim=-1)


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


def pop_only(elements: Iterable[X]) -> X:
    """Unpack a one element list, or raise an error."""
    elements = tuple(elements)
    if len(elements) == 0:
        raise ValueError('Empty sequence given')
    if len(elements) > 1:
        raise ValueError(f'More than one element: {elements}')
    return elements[0]


def strip_dim(*x, num: int = 4):
    """Strip the first dimensions."""
    return [xx.view(xx.shape[num:]) for xx in x]


def upgrade_to_sequence(x: Union[X, Sequence[X]]) -> Sequence[X]:
    """Ensure that the input is a sequence."""
    return x if isinstance(x, Sequence) else (x,)
