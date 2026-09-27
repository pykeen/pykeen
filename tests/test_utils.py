"""Tests for the :mod:`pykeen.utils` module."""

import contextlib
import functools
import io
import itertools
import operator
import os
import pathlib
import random
import string
import tempfile
import timeit
import unittest
from collections.abc import Iterable
from typing import Any
from unittest import mock

import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.csgraph
import scipy.special
import torch

from pykeen.utils import (
    _weisfeiler_lehman_iteration,
    _weisfeiler_lehman_iteration_approx,
    calculate_broadcasted_elementwise_result_shape,
    clamp_norm,
    combine_complex,
    compact_mapping,
    compose,
    estimate_cost_of_sequence,
    find,
    flatten_dictionary,
    get_connected_components,
    get_optimal_sequence,
    get_until_first_blank,
    iter_weisfeiler_lehman,
    logcumsumexp,
    merge_kwargs,
    normalize_path,
    project_entity,
    resolve_device,
    set_random_seed,
    split_complex,
    tensor_product,
    tensor_sum,
    view_complex,
    view_complex_native,
)


class TestCompose(unittest.TestCase):
    """Tests for composition."""

    def test_compose(self):
        """Test composition."""

        def _f(x):
            return x + 2

        def _g(x):
            return 2 * x

        fog = compose(_f, _g, name="fog")
        for i in range(5):
            with self.subTest(i=i):
                assert _g(_f(i)) == fog(i)
                assert _g(_f(i**2)) == fog(i**2)


class FlattenDictionaryTest(unittest.TestCase):
    """Test flatten_dictionary."""

    def test_flatten_dictionary(self):
        """Test if the output of flatten_dictionary is correct."""
        nested_dictionary = {
            "a": {
                "b": {
                    "c": 1,
                    "d": 2,
                },
                "e": 3,
            },
        }
        expected_output = {
            "a.b.c": 1,
            "a.b.d": 2,
            "a.e": 3,
        }
        observed_output = flatten_dictionary(nested_dictionary)
        self._compare(observed_output, expected_output)

    def test_flatten_dictionary_mixed_key_type(self):
        """Test if the output of flatten_dictionary is correct if some keys are not strings."""
        nested_dictionary = {
            "a": {
                5: {
                    "c": 1,
                    "d": 2,
                },
                "e": 3,
            },
        }
        expected_output = {
            "a.5.c": 1,
            "a.5.d": 2,
            "a.e": 3,
        }
        observed_output = flatten_dictionary(nested_dictionary)
        self._compare(observed_output, expected_output)

    def test_flatten_dictionary_prefix(self):
        """Test if the output of flatten_dictionary is correct."""
        nested_dictionary = {
            "a": {
                "b": {
                    "c": 1,
                    "d": 2,
                },
                "e": 3,
            },
        }
        expected_output = {
            "Test.a.b.c": 1,
            "Test.a.b.d": 2,
            "Test.a.e": 3,
        }
        observed_output = flatten_dictionary(nested_dictionary, prefix="Test")
        self._compare(observed_output, expected_output)

    def _compare(self, observed_output, expected_output):
        assert not any(isinstance(o, dict) for o in expected_output.values())
        assert expected_output == observed_output


class TestGetUntilFirstBlank(unittest.TestCase):
    """Test get_until_first_blank()."""

    def test_get_until_first_blank_trivial(self):
        """Test the trivial string."""
        s = ""
        r = get_until_first_blank(s)
        assert r == ""

    def test_regular(self):
        """Test a regulat case."""
        s = """Broken
        line.

        Now I continue.
        """
        r = get_until_first_blank(s)
        assert r == "Broken line."

    def test_multi_line_first_paragraph(self):
        """Test a first paragraph spanning more than two lines."""
        assert get_until_first_blank("A\nB\nC\n\nD") == "A B C"

    def test_whitespace_only_blank_line(self):
        """Test that a line consisting of whitespace only counts as blank."""
        assert get_until_first_blank("A\n  B\n  C\n    \n  D") == "A B C"

    def test_no_blank_line(self):
        """Test a string without any blank line."""
        assert get_until_first_blank("A\n  B\n  C") == "A B C"

    def test_leading_blank_lines(self):
        """Test that leading blank lines are skipped."""
        assert get_until_first_blank("\n   \n  A\n  B\n  C\n\n  D") == "A B C"

    def test_class_docstring(self):
        """Test a typical indented class docstring."""
        doc = """
            A model with a summary
            spanning several lines.

            Some more details, which are not part of the summary.

            ---
            name: A
        """
        assert get_until_first_blank(doc) == "A model with a summary spanning several lines."


def _generate_shapes(
    n_dim: int = 5,
    n_terms: int = 4,
    iterations: int = 64,
    *,
    generator: torch.Generator,
) -> Iterable[tuple[tuple[int, ...], ...]]:
    """Generate shapes."""
    max_shape = torch.randint(low=2, high=32, size=(128,), generator=generator)
    for _ in range(iterations):
        # create broadcastable shapes
        idx = torch.randperm(max_shape.shape[0], generator=generator)[:n_dim]
        this_max_shape = max_shape[idx]
        this_min_shape = torch.ones_like(this_max_shape)
        shapes = []
        for _j in range(n_terms):
            mask = this_min_shape
            while not (1 < mask.sum() < n_dim):
                mask = torch.as_tensor(torch.rand(size=(n_dim,), generator=generator) < 0.3, dtype=max_shape.dtype)
            this_array_shape = this_max_shape * mask + this_min_shape * (1 - mask)
            shapes.append(tuple(this_array_shape.tolist()))
        yield tuple(shapes)


class TestUtils(unittest.TestCase):
    """Tests for :mod:`pykeen.utils`."""

    def test_compact_mapping(self):
        """Test ``compact_mapping()``."""
        mapping = {letter: 2 * i for i, letter in enumerate(string.ascii_letters)}
        compacted_mapping, id_remapping = compact_mapping(mapping=mapping)

        # check correct value range
        assert set(compacted_mapping.values()) == set(range(len(mapping)))
        assert set(id_remapping.keys()) == set(mapping.values())
        assert set(id_remapping.values()) == set(compacted_mapping.values())

    def test_clamp_norm(self):
        """Test clamp_norm() ."""
        max_norm = 1.0
        gen = torch.manual_seed(42)
        eps = 1.0e-06
        for p in [1, 2, float("inf")]:
            for _ in range(10):
                x = torch.rand(10, 20, 30, generator=gen)
                for dim in range(x.ndimension()):
                    x_c = clamp_norm(x, maxnorm=max_norm, p=p, dim=dim)

                    # check maximum norm constraint
                    assert (x_c.norm(p=p, dim=dim) <= max_norm + eps).all()

                    # unchanged values for small norms
                    norm = x.norm(p=p, dim=dim)
                    mask = torch.stack([(norm < max_norm)] * x.shape[dim], dim=dim)
                    assert (x_c[mask] == x[mask]).all()

    def test_complex_utils(self):
        """Test complex tensor utilities."""
        re = torch.rand(20, 10)
        im = torch.rand(20, 10)
        x = combine_complex(x_re=re, x_im=im)
        re2, im2 = split_complex(x)
        assert (re2 == re).all()
        assert (im2 == im).all()

    def test_project_entity(self):
        """Test _project_entity."""
        batch_size = 2
        embedding_dim = 3
        relation_dim = 5
        num_entities = 7

        # random entity embeddings & projections
        e = torch.rand(1, num_entities, embedding_dim)
        e = clamp_norm(e, maxnorm=1, p=2, dim=-1)
        e_p = torch.rand(1, num_entities, embedding_dim)

        # random relation embeddings & projections
        r_p = torch.rand(batch_size, 1, relation_dim)

        # project
        e_bot = project_entity(e=e, e_p=e_p, r_p=r_p)

        # check shape:
        assert e_bot.shape == (batch_size, num_entities, relation_dim)

        # check normalization
        assert (torch.norm(e_bot, dim=-1, p=2) <= 1.0 + 1.0e-06).all()

        # check equivalence of re-formulation
        # e_{\bot} = M_{re} e = (r_p e_p^T + I^{d_r \times d_e}) e
        #                     = r_p (e_p^T e) + e'
        m_re = r_p.unsqueeze(dim=-1) @ e_p.unsqueeze(dim=-2)
        m_re = m_re + torch.eye(relation_dim, embedding_dim).view(1, 1, relation_dim, embedding_dim)
        assert m_re.shape == (batch_size, num_entities, relation_dim, embedding_dim)
        e_vanilla = (m_re @ e.unsqueeze(dim=-1)).squeeze(dim=-1)
        e_vanilla = clamp_norm(e_vanilla, p=2, dim=-1, maxnorm=1)
        assert torch.allclose(e_vanilla, e_bot)

    def test_calculate_broadcasted_elementwise_result_shape(self):
        """Test calculate_broadcasted_elementwise_result_shape."""
        max_dim = 64
        for n_dim, _ in itertools.product(range(2, 5), range(10)):
            a_shape = [1 for _ in range(n_dim)]
            b_shape = [1 for _ in range(n_dim)]
            for j in range(n_dim):
                dim = 2 + random.randrange(max_dim)  # noqa:S311
                mod = random.randrange(3)  # noqa:S311
                if mod % 2 == 0:
                    a_shape[j] = dim
                if mod > 0:
                    b_shape[j] = dim
                a = torch.empty(*a_shape)
                b = torch.empty(*b_shape)
                shape = calculate_broadcasted_elementwise_result_shape(first=a.shape, second=b.shape)
                c = a + b
                exp_shape = c.shape
                assert shape == exp_shape

    @unittest.skip("This is often failing non-deterministically")
    def test_estimate_cost_of_add_sequence(self):
        """Test ``estimate_cost_of_add_sequence()``."""
        _, generator, _ = set_random_seed(seed=42)
        # create random array, estimate the costs of addition, and measure some execution times.
        # then, compute correlation between the estimated cost, and the measured time.
        data = []
        for shapes in _generate_shapes(generator=generator):
            arrays = [torch.empty(*shape) for shape in shapes]
            cost = estimate_cost_of_sequence(*(a.shape for a in arrays))
            n_samples, time = timeit.Timer(stmt="sum(arrays)", globals={"arrays": arrays}).autorange()
            consumption = time / n_samples
            data.append((cost, consumption))
        a = np.asarray(data)

        # check for strong correlation between estimated costs and measured execution time
        assert (np.corrcoef(x=a[:, 0], y=a[:, 1])[0, 1]) > 0.8

    @pytest.mark.slow
    def test_get_optimal_sequence_caching(self):
        """Test caching of ``get_optimal_sequence()``."""
        _, generator, _ = set_random_seed(seed=42)
        for shapes in _generate_shapes(iterations=10, generator=generator):
            # get optimal sequence
            first_time = timeit.default_timer()
            get_optimal_sequence(*shapes)
            first_time = timeit.default_timer() - first_time

            # check caching
            samples, second_time = timeit.Timer(
                stmt="get_optimal_sequence(*shapes)",
                globals={
                    "get_optimal_sequence": get_optimal_sequence,
                    "shapes": shapes,
                },
            ).autorange()
            second_time /= samples

            assert second_time < first_time

    def test_get_optimal_sequence(self):
        """Test ``get_optimal_sequence()``."""
        _, generator, _ = set_random_seed(seed=42)
        for shapes in _generate_shapes(generator=generator):
            # get optimal sequence
            opt_cost, opt_seq = get_optimal_sequence(*shapes)

            # check correct cost
            exp_opt_cost = estimate_cost_of_sequence(*(shapes[i] for i in opt_seq))
            assert exp_opt_cost == opt_cost

            # check optimality
            for perm in itertools.permutations(list(range(len(shapes)))):
                cost = estimate_cost_of_sequence(*(shapes[i] for i in perm))
                assert cost >= opt_cost

    def test_tensor_sum(self):
        """Test tensor_sum."""
        _, generator, _ = set_random_seed(seed=42)
        for shapes in _generate_shapes(generator=generator):
            tensors = [torch.rand(*shape) for shape in shapes]
            result = tensor_sum(*tensors)

            # compare result to sequential addition
            assert torch.allclose(result, sum(tensors))

    def test_tensor_product(self):
        """Test tensor_product."""
        _, generator, _ = set_random_seed(seed=42)
        for shapes in _generate_shapes(generator=generator):
            tensors = [torch.rand(*shape) for shape in shapes]
            result = tensor_product(*tensors)

            # compare result to sequential addition
            assert torch.allclose(result, functools.reduce(operator.mul, tensors[1:], tensors[0]))

    def test_weisfeiler_lehman(self):
        """Test Weisfeiler Lehman."""
        _, generator, _ = set_random_seed(seed=42)
        num_nodes = 13
        num_edges = 31
        max_iter = 3
        edge_index = torch.randint(num_nodes, size=(2, num_edges), generator=generator)
        # ensure each node participates in at least one edge
        edge_index[0, :num_nodes] = torch.arange(num_nodes)

        count = 0
        color_count = 0
        for colors in iter_weisfeiler_lehman(edge_index=edge_index, max_iter=max_iter):
            # check type and shape
            assert torch.is_tensor(colors)
            assert colors.shape == (num_nodes,)
            assert colors.dtype == torch.long
            # number of colors is monotonically increasing
            num_unique_colors = len(colors.unique())
            assert num_unique_colors >= color_count
            color_count = num_unique_colors
            count += 1
        assert count == max_iter

    def test_weisfeiler_lehman_approximation(self):
        """Verify approximate WL."""
        _, generator, _ = set_random_seed(seed=42)
        num_nodes = 13
        num_edges = 31
        edge_index = torch.randint(num_nodes, size=(2, num_edges), generator=generator)
        # ensure each node participates in at least one edge
        edge_index[0, :num_nodes] = torch.arange(num_nodes)
        edge_index = edge_index.unique(dim=1)
        adj = torch.sparse_coo_tensor(indices=edge_index, values=torch.ones(size=edge_index[0].shape))
        colors = torch.randint(3, size=(num_nodes,))
        reference = _weisfeiler_lehman_iteration(adj=adj, colors=colors)
        approx = _weisfeiler_lehman_iteration_approx(adj=adj, colors=colors, dim=4)
        # normalize
        sim_ref = reference[None, :] == reference[:, None]
        sim_approx = approx[None, :] == approx[:, None]
        assert torch.allclose(sim_ref, sim_approx)


@pytest.mark.parametrize(
    ("kwargs", "extra", "expected", "error"),
    [
        ({"max_id": 10}, {}, {"max_id": 10}, None),
        ({}, {"max_id": 10}, {"max_id": 10}, None),
        ({"max_id": 10}, {"max_id": None}, {"max_id": 10}, None),
        ({"max_id": 10}, {"max_id": 7}, ..., ValueError),
        (
            [{"shape": (3,)}, {"shape": (4,)}],
            {"max_id": 7},
            [{"shape": (3,), "max_id": 7}, {"shape": (4,), "max_id": 7}],
            None,
        ),
    ],
)
def test_merge_kwargs(
    kwargs: dict[str, Any], extra: dict[str, Any], expected: dict[str, Any], error: type[BaseException] | None
) -> None:
    """Test merging of parameters."""
    with pytest.raises(error) if error else contextlib.nullcontext():
        merged_kwargs = merge_kwargs(kwargs=kwargs, **extra)
        assert merged_kwargs == expected


@pytest.mark.parametrize(
    ("device", "cuda_available", "mps_available", "expected"),
    [
        (None, False, False, "cpu"),
        (None, False, True, "mps"),
        (None, True, False, "cuda"),
        (None, True, True, "cuda"),
        ("gpu", False, False, "cpu"),
        ("gpu", False, True, "mps"),
        ("gpu", True, False, "cuda"),
        ("cuda", False, False, "cpu"),
        ("cuda", False, True, "mps"),
        ("cuda", True, False, "cuda"),
        ("mps", False, False, "cpu"),
        ("mps", False, True, "mps"),
        ("cpu", False, False, "cpu"),
        ("cpu", True, True, "cpu"),
    ],
)
def test_resolve_device(device: str | None, cuda_available: bool, mps_available: bool, expected: str) -> None:
    """Test device resolution with (un)available accelerators."""
    with (
        mock.patch("torch.cuda.is_available", return_value=cuda_available),
        mock.patch("torch.backends.mps.is_available", return_value=mps_available),
    ):
        assert resolve_device(device).type == expected


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        # empty graph
        ([], []),
        # single edge
        ([(1, 2)], [[1, 2]]),
        # two disjoint components
        ([(1, 2), (3, 4)], [[1, 2], [3, 4]]),
        # the last edge merges two components, leaving node 1 as root only reachable via an intermediate node
        ([(1, 2), (3, 4), (2, 3)], [[1, 2, 3, 4]]),
        # chain in reverse order
        ([(4, 3), (3, 2), (2, 1)], [[1, 2, 3, 4]]),
        # self-loop and cycle
        ([(1, 1), (2, 3), (3, 5), (5, 2)], [[1], [2, 3, 5]]),
    ],
)
def test_get_connected_components(pairs: list[tuple[int, int]], expected: list[list[int]]) -> None:
    """Test calculation of connected components."""
    components = get_connected_components(pairs)
    assert sorted(sorted(component) for component in components) == expected


def _normalize_components(components: Iterable[Iterable[int]]) -> set[frozenset[int]]:
    """Convert components to a set of frozensets, and verify that there are no duplicates."""
    components = [list(component) for component in components]
    result = {frozenset(component) for component in components}
    # no duplicate nodes, neither within nor across components
    assert sum(map(len, components)) == sum(map(len, result))
    assert len(result) == len(components)
    return result


def _reference_connected_components(pairs: list[tuple[int, int]]) -> set[frozenset[int]]:
    """Calculate connected components with scipy."""
    nodes = sorted({node for pair in pairs for node in pair})
    node_to_id = {node: i for i, node in enumerate(nodes)}
    row, col = np.asarray([(node_to_id[x], node_to_id[y]) for x, y in pairs]).T
    matrix = scipy.sparse.coo_matrix((np.ones_like(row), (row, col)), shape=(len(nodes), len(nodes)))
    _, labels = scipy.sparse.csgraph.connected_components(matrix, directed=False)
    result: dict[int, set[int]] = {}
    for node, label in zip(nodes, labels, strict=True):
        result.setdefault(label, set()).add(node)
    return {frozenset(component) for component in result.values()}


@pytest.mark.parametrize("seed", range(10))
def test_get_connected_components_random(seed: int) -> None:
    """Compare get_connected_components against scipy on random graphs."""
    generator = np.random.default_rng(seed=seed)
    num_nodes = int(generator.integers(2, 50))
    num_edges = int(generator.integers(1, 2 * num_nodes))
    pairs = [(int(x), int(y)) for x, y in generator.integers(num_nodes, size=(num_edges, 2))]
    assert _normalize_components(get_connected_components(pairs)) == _reference_connected_components(pairs)


def test_find_path_compression() -> None:
    """Test that find compresses the path to the root."""
    # chain 4 -> 3 -> 2 -> 1 -> 0
    parent = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
    assert find(x=4, parent=parent) == 0
    assert parent == dict.fromkeys(range(5), 0)
    with pytest.raises(ValueError, match="Unknown element"):
        find(x=5, parent=parent)
class LogCumSumExpTests(unittest.TestCase):
    """Tests for :func:`pykeen.utils.logcumsumexp`."""

    def setUp(self) -> None:
        """Set up the random number generator."""
        self.generator = numpy.random.default_rng(seed=42)

    def test_large_spread(self):
        """Test that small prefixes do not underflow when a much larger value follows."""
        numpy.testing.assert_allclose(logcumsumexp(numpy.asarray([-1000.0, 0.0])), [-1000.0, 0.0])

    def test_prefix_logsumexp(self):
        """Test agreement with :func:`scipy.special.logsumexp` on prefixes."""
        for scale, ascending in itertools.product((1.0, 100.0, 1000.0), (False, True)):
            a = scale * self.generator.normal(size=(17,))
            if ascending:
                # early prefixes are far below the global maximum
                a = numpy.sort(a)
            result = logcumsumexp(a)
            expected = numpy.asarray([scipy.special.logsumexp(a[: i + 1]) for i in range(len(a))])
            with self.subTest(scale=scale, ascending=ascending):
                assert numpy.isfinite(result).all()
                numpy.testing.assert_allclose(result, expected)

    def test_all_neg_inf(self):
        """Test that all ``-inf`` input yields ``-inf`` output (and not nan)."""
        numpy.testing.assert_array_equal(logcumsumexp(numpy.full(shape=(3,), fill_value=-numpy.inf)), -numpy.inf)

    def test_shape(self):
        """Test the output shape for ND input."""
        a = self.generator.normal(size=(2, 3, 4))
        # default: flatten, like numpy.cumsum
        numpy.testing.assert_allclose(logcumsumexp(a), logcumsumexp(a.ravel()))
        assert logcumsumexp(a).shape == (a.size,)
        for axis in (0, 1, 2, -1):
            with self.subTest(axis=axis):
                result = logcumsumexp(a, axis=axis)
                assert result.shape == a.shape
                expected = numpy.log(numpy.cumsum(numpy.exp(a), axis=axis))
                numpy.testing.assert_allclose(result, expected)

    def test_torch(self):
        """Test agreement with :func:`torch.logcumsumexp`."""
        a = 100.0 * self.generator.normal(size=(5, 7))
        for axis in (0, 1):
            with self.subTest(axis=axis):
                expected = torch.logcumsumexp(torch.as_tensor(a), dim=axis).numpy()
                numpy.testing.assert_allclose(logcumsumexp(a, axis=axis), expected)
class TestNormalizePath(unittest.TestCase):
    """Tests for :func:`pykeen.utils.normalize_path`."""

    def setUp(self) -> None:
        """Create a temporary directory with a file."""
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = pathlib.Path(self._tmp.name).resolve()
        self.file_path = self.directory.joinpath("file.txt")
        self.file_path.write_text("hello")

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        self._tmp.cleanup()

    def test_str(self) -> None:
        """Test normalizing a string path."""
        assert normalize_path(str(self.file_path)) == self.file_path

    def test_text_file_handle(self) -> None:
        """Test normalizing a text-mode file handle."""
        with self.file_path.open() as file:
            assert normalize_path(file) == self.file_path

    def test_binary_file_handle(self) -> None:
        """Test normalizing a binary-mode file handle."""
        with self.file_path.open("rb") as file:
            assert normalize_path(file) == self.file_path

    def test_file_handle_other(self) -> None:
        """Test normalizing a file handle with additional parts."""
        with self.file_path.open() as file:
            assert normalize_path(file, "a", "b") == self.file_path.joinpath("a", "b")

    def test_file_handle_mkdir_is_file(self) -> None:
        """Test normalizing a file handle with ``mkdir=True, is_file=True``, which must not touch the file."""
        with self.file_path.open() as file:
            assert normalize_path(file, mkdir=True, is_file=True) == self.file_path
        assert self.file_path.is_file()

    def test_file_handle_as_default(self) -> None:
        """Test using a file handle as default."""
        with self.file_path.open() as file:
            assert normalize_path(None, default=file) == self.file_path

    def test_in_memory_buffer(self) -> None:
        """Test that in-memory buffers without a file name raise a clear error."""
        for buffer in (io.StringIO("hello"), io.BytesIO(b"hello")):
            with self.subTest(buffer=type(buffer)), pytest.raises(TypeError, match="file handle"):
                normalize_path(buffer)

    def test_file_descriptor_handle(self) -> None:
        """Test that handles opened from a file descriptor (with an integer name) raise a clear error."""
        fd = os.open(self.file_path, os.O_RDONLY)
        with os.fdopen(fd) as file, pytest.raises(TypeError, match="file handle"):
            normalize_path(file)
@pytest.mark.parametrize("shape", [(6,), (3, 4), (2, 3, 8)])
def test_view_complex(shape: tuple[int, ...]) -> None:
    """Test converting real-valued tensors with interleaved real/imaginary parts to complex ones."""
    x = torch.rand(*shape)
    y = view_complex(x)
    assert y.is_complex()
    assert y.shape == (*shape[:-1], shape[-1] // 2)
    # round-trip
    assert torch.equal(torch.view_as_real(y).view(x.shape), x)
    # consistency with native implementation
    assert torch.equal(y, view_complex_native(x))
    # interleaved layout
    assert torch.equal(y.real, x[..., 0::2])
    assert torch.equal(y.imag, x[..., 1::2])


def test_view_complex_complex_input() -> None:
    """Test that complex input is passed through unchanged."""
    x = torch.rand(3, 4, dtype=torch.cfloat)
    assert view_complex(x) is x


def test_view_complex_non_contiguous() -> None:
    """Test conversion of non-contiguous input."""
    base = torch.rand(6, 5)
    x = base.t()
    assert not x.is_contiguous()
    y = view_complex(x)
    assert y.shape == (5, 3)
    assert torch.equal(y, view_complex_native(x.contiguous()))
    assert torch.equal(torch.view_as_real(y).reshape(x.shape), x)


def test_view_complex_odd_dimension() -> None:
    """Test that an odd last dimension raises an error."""
    with pytest.raises(ValueError, match="even"):
        view_complex(torch.rand(2, 5))
