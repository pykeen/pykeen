# -*- coding: utf-8 -*-

"""Tests for the :mod:`pykeen.utils` module."""

import functools
import itertools
import operator
import random
import string
import timeit
import unittest
from typing import Iterable, Tuple

import numpy
import pytest
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
    flatten_dictionary,
    get_optimal_sequence,
    get_until_first_blank,
    iter_weisfeiler_lehman,
    logcumsumexp,
    project_entity,
    set_random_seed,
    split_complex,
    tensor_product,
    tensor_sum,
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
                self.assertEqual(_g(_f(i)), fog(i))
                self.assertEqual(_g(_f(i**2)), fog(i**2))


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
        self.assertEqual("", r)

    def test_regular(self):
        """Test a regulat case."""
        s = """Broken
        line.

        Now I continue.
        """
        r = get_until_first_blank(s)
        self.assertEqual("Broken line.", r)


def _generate_shapes(
    n_dim: int = 5,
    n_terms: int = 4,
    iterations: int = 64,
    *,
    generator: torch.Generator,
) -> Iterable[Tuple[Tuple[int, ...], ...]]:
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
        self.assertEqual(set(compacted_mapping.values()), set(range(len(mapping))))
        self.assertEqual(set(id_remapping.keys()), set(mapping.values()))
        self.assertEqual(set(id_remapping.values()), set(compacted_mapping.values()))

    def test_clamp_norm(self):
        """Test  clamp_norm() ."""
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
                dim = 2 + random.randrange(max_dim)
                mod = random.randrange(3)
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
            n_samples, time = timeit.Timer(stmt="sum(arrays)", globals=dict(arrays=arrays)).autorange()
            consumption = time / n_samples
            data.append((cost, consumption))
        a = numpy.asarray(data)

        # check for strong correlation between estimated costs and measured execution time
        assert (numpy.corrcoef(x=a[:, 0], y=a[:, 1])[0, 1]) > 0.8

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
                globals=dict(
                    get_optimal_sequence=get_optimal_sequence,
                    shapes=shapes,
                ),
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

    def test_logcumsumexp(self):
        """Verify that our numpy implementation gives the same results as the torch variant."""
        generator = numpy.random.default_rng(seed=42)
        a = generator.random(size=(21,))
        r1 = logcumsumexp(a)
        r2 = torch.logcumsumexp(torch.as_tensor(a), dim=0).numpy()
        numpy.testing.assert_allclose(r1, r2)

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
