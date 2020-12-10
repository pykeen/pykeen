# -*- coding: utf-8 -*-

"""Unittest for the :mod:`pykeen.nn` module."""

import itertools
import unittest
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence
from unittest.mock import MagicMock, Mock

import numpy
import pytest
import torch
from torch.nn import functional

from pykeen.nn import Embedding, EmbeddingSpecification, LiteralRepresentations, RepresentationModule
from pykeen.nn.representation import (
    CANONICAL_DIMENSIONS, RGCNRepresentations, convert_to_canonical_shape, get_expected_canonical_shape,
)
from pykeen.nn.sim import _torch_kl_similarity, kullback_leibler_similarity
from pykeen.testing import base as ptb
from pykeen.testing.mocks import MockRepresentations
from pykeen.triples import TriplesFactory
from pykeen.typing import GaussianDistribution


class RepresentationModuleTests(ptb.GenericTests[RepresentationModule]):
    """Tests for RepresentationModule."""

    #: The batch size
    batch_size: int = 3

    #: The number of representations
    num: int = 5

    #: The expected shape of an individual representation
    exp_shape: Sequence[int] = (5,)

    def post_instantiation_hook(self) -> None:  # noqa: D102
        self.instance.reset_parameters()

    def test_max_id(self):
        """Test the maximum ID."""
        assert self.instance.max_id == self.num

    def test_shape(self):
        """Test the shape."""
        assert self.instance.shape == self.exp_shape

    def _test_forward(self, indices: Optional[torch.LongTensor]):
        """Test the forward method."""
        x = self.instance(indices=indices)
        assert torch.is_tensor(x)
        assert x.dtype == torch.float32
        n = self.num if indices is None else indices.shape[0]
        assert x.shape == tuple([n, *self.instance.shape])
        self._verify_content(x=x, indices=indices)

    def _verify_content(self, x, indices):
        """Additional verification."""
        assert x.requires_grad

    def _valid_indices(self) -> Iterable[torch.LongTensor]:
        return [
            torch.randint(self.num, size=(self.batch_size,)),
            torch.randperm(self.num),
            torch.randperm(self.num).repeat(2),
        ]

    def _invalid_indices(self) -> Iterable[torch.LongTensor]:
        return [
            torch.as_tensor([self.num], dtype=torch.long),  # too high index
            torch.randint(self.num, size=(2, 3)),  # too many indices
        ]

    def test_forward_without_indices(self):
        """Test forward without providing indices."""
        self._test_forward(indices=None)

    def test_forward_with_indices(self):
        """Test forward with providing indices."""
        for indices in self._valid_indices():
            self._test_forward(indices=indices)

    def test_forward_with_invalid_indices(self):
        """Test whether passing invalid indices crashes."""
        for indices in self._invalid_indices():
            with pytest.raises((IndexError, RuntimeError)):
                self._test_forward(indices=indices)

    def _test_in_canonical_shape(self, indices: Optional[torch.LongTensor]):
        """Test get_in_canonical_shape with the given indices."""
        # test both, using the actual dimension, and its name
        for dim in itertools.chain(CANONICAL_DIMENSIONS.keys(), CANONICAL_DIMENSIONS.values()):
            # batch_size, d1, d2, d3, *
            x = self.instance.get_in_canonical_shape(dim=dim, indices=indices)

            # data type
            assert torch.is_tensor(x)
            assert x.dtype == torch.float32  # todo: adjust?
            assert x.ndimension() == 4 + len(self.exp_shape)

            # get expected shape
            exp_shape = get_expected_canonical_shape(
                indices=indices,
                dim=dim,
                suffix_shape=self.exp_shape,
                num=self.num,
            )
            assert x.shape == exp_shape

    def test_get_in_canonical_shape_without_indices(self):
        """Test get_in_canonical_shape without indices, i.e. with 1-n scoring."""
        self._test_in_canonical_shape(indices=None)

    def test_get_in_canonical_shape_with_indices(self):
        """Test get_in_canonical_shape with 1-dimensional indices."""
        for indices in self._valid_indices():
            self._test_in_canonical_shape(indices=indices)

    def test_get_in_canonical_shape_with_2d_indices(self):
        """Test get_in_canonical_shape with 2-dimensional indices."""
        indices = torch.randint(self.num, size=(self.batch_size, 2))
        self._test_in_canonical_shape(indices=indices)


def _check_call(
    self: unittest.TestCase,
    call_count: int,
    should_be_called: bool,
    wrapped: MagicMock,
    kwargs: Optional[Mapping[str, Any]],
) -> int:
    """
    Check whether a wrapped method is called.

    :param self:
        The test cas calling the check
    :param call_count:
        The previous call count.
    :param should_be_called:
        Whether it should be called.
    :param wrapped:
        The wrapped method.
    :param kwargs:
        The expected kwargs when called.

    :return:
        The updated counter.
    """
    if should_be_called:
        call_count += 1

        self.assertEqual(call_count, wrapped.call_count)

        # Lets check the tuple
        self.assertIsInstance(wrapped.call_args, tuple)

        call_size = len(wrapped.call_args)
        # Make sure tuple at least has positional arguments, could be 3 if kwargs available
        self.assertLessEqual(2, call_size)

        if call_size == 2:
            args_idx, kwargs_idx = 0, 1
        else:  # call_size == 3:
            args_idx, kwargs_idx = 1, 2

        # called with one positional argument ...
        self.assertEqual(1, len(wrapped.call_args[args_idx]),
                         msg=f'Args: {wrapped.call_args[args_idx]} Kwargs: {wrapped.call_args[kwargs_idx]}')
        # .. and additional key-word based arguments.
        self.assertEqual(len(kwargs or {}), len(wrapped.call_args[kwargs_idx]))
    else:
        self.assertEqual(call_count, wrapped.call_count)
    return call_count


class EmbeddingTests(RepresentationModuleTests, unittest.TestCase):
    """Tests for Embedding."""

    cls = Embedding
    kwargs = dict(
        num_embeddings=RepresentationModuleTests.num,
        shape=RepresentationModuleTests.exp_shape,
    )

    def test_constructor_errors(self):
        """Test error cases for constructor call."""
        for embedding_dim, shape in (
            (None, None),  # neither
            (3, (5, 3)),  # both
        ):
            with pytest.raises(ValueError):
                Embedding(
                    num_embeddings=self.num,
                    embedding_dim=embedding_dim,
                    shape=shape,
                )

    def _test_func_with_kwargs(
        self,
        name: str,
        func,
        kwargs: Optional[Mapping[str, Any]] = None,
        reset_parameters_call: bool = False,
        forward_call: bool = False,
        post_parameter_update_call: bool = False,
    ):
        """Test initializer usage."""
        # wrap to check calls
        wrapped = MagicMock(side_effect=func)

        # instantiate embedding
        embedding_kwargs = {name: wrapped}
        if kwargs is not None:
            embedding_kwargs[f"{name}_kwargs"] = kwargs
        embedding = Embedding(
            num_embeddings=self.num,
            shape=self.exp_shape,
            **embedding_kwargs,
        )

        # check that nothing gets called in constructor
        wrapped.assert_not_called()
        call_count = 0

        # check call in reset_parameters
        embedding.reset_parameters()
        call_count = _check_call(
            self,
            call_count=call_count,
            should_be_called=reset_parameters_call,
            wrapped=wrapped,
            kwargs=kwargs,
        )

        # check call in forward
        embedding.forward(indices=None)
        call_count = _check_call(
            self,
            call_count=call_count,
            should_be_called=forward_call,
            wrapped=wrapped,
            kwargs=kwargs,
        )

        # check call in post_parameter_update
        embedding.post_parameter_update()
        _check_call(
            self,
            call_count=call_count,
            should_be_called=post_parameter_update_call,
            wrapped=wrapped,
            kwargs=kwargs,
        )

    def test_initializer(self):
        """Test initializer."""
        self._test_func_with_kwargs(
            name="initializer",
            func=torch.nn.init.normal_,
            reset_parameters_call=True,
        )

    def test_initializer_with_kwargs(self):
        """Test initializer with kwargs."""
        self._test_func_with_kwargs(
            name="initializer",
            func=torch.nn.init.normal_,
            kwargs=dict(mean=3),
            reset_parameters_call=True,
        )

    def test_normalizer(self):
        """Test normalizer."""
        self._test_func_with_kwargs(
            name="normalizer",
            func=functional.normalize,
            forward_call=True,
        )

    def test_normalizer_kwargs(self):
        """Test normalizer with kwargs."""
        self._test_func_with_kwargs(
            name="normalizer",
            func=functional.normalize,
            kwargs=dict(p=1),
            forward_call=True,
        )

    def test_constrainer(self):
        """Test constrainer."""
        self._test_func_with_kwargs(
            name="constrainer",
            func=functional.normalize,
            post_parameter_update_call=True,
        )

    def test_constrainer_kwargs(self):
        """Test constrainer with kwargs."""
        self._test_func_with_kwargs(
            name="constrainer",
            func=functional.normalize,
            kwargs=dict(p=1),
            post_parameter_update_call=True,
        )


class TensorEmbeddingTests(RepresentationModuleTests, unittest.TestCase):
    """Tests for Embedding with 2-dimensional shape."""

    cls = Embedding
    exp_shape = (3, 7)
    kwargs = dict(
        num_embeddings=RepresentationModuleTests.num,
        shape=(3, 7),
    )


class LiteralRepresentationsTests(EmbeddingTests, unittest.TestCase):
    """Tests for literal embeddings."""

    cls = LiteralRepresentations

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        # requires own kwargs
        kwargs.clear()
        self.numeric_literals = torch.rand(self.num, *self.exp_shape)
        kwargs["numeric_literals"] = self.numeric_literals
        return kwargs

    def _verify_content(self, x, indices):  # noqa: D102
        exp_x = self.numeric_literals
        if indices is not None:
            exp_x = exp_x[indices]
        self.assertTrue(torch.allclose(x, exp_x))


class RGCNRepresentationTests(RepresentationModuleTests, unittest.TestCase):
    """Test RGCN representations."""

    cls = RGCNRepresentations
    kwargs = dict(
        num_bases_or_blocks=2,
        embedding_dim=RepresentationModuleTests.exp_shape[0],
    )
    num_relations: int = 7
    num_triples: int = 31
    num_bases: int = 2

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        # TODO: use triple generation
        # generate random triples
        mapped_triples = numpy.stack([
            numpy.random.randint(max_id, size=(self.num_triples,))
            for max_id in (self.num, self.num_relations, self.num)
        ], axis=-1)
        entity_names = [f"e_{i}" for i in range(self.num)]
        relation_names = [f"r_{i}" for i in range(self.num_relations)]
        triples = numpy.stack([
            [names[i] for i in col.tolist()]
            for col, names in zip(
                mapped_triples.T,
                (entity_names, relation_names, entity_names),
            )
        ])
        kwargs["triples_factory"] = TriplesFactory.from_labeled_triples(triples=triples)
        return kwargs


class RepresentationModuleTestsTest(ptb.TestsTest[RepresentationModule], unittest.TestCase):
    """Test that there are tests for all representation modules."""

    base_cls = RepresentationModule
    base_test = RepresentationModuleTests
    skip_cls = {MockRepresentations}


class EmbeddingSpecificationTests(unittest.TestCase):
    """Tests for EmbeddingSpecification."""

    #: The number of embeddings
    num: int = 3

    def test_make(self):
        """Test make."""
        initializer = Mock()
        normalizer = Mock()
        constrainer = Mock()
        regularizer = Mock()
        for embedding_dim, shape in [
            (None, (3,)),
            (None, (3, 5)),
            (3, None),
        ]:
            spec = EmbeddingSpecification(
                embedding_dim=embedding_dim,
                shape=shape,
                initializer=initializer,
                normalizer=normalizer,
                constrainer=constrainer,
                regularizer=regularizer,
            )
            emb = spec.make(num_embeddings=self.num)

            # check shape
            self.assertEqual(emb.embedding_dim, (embedding_dim or int(numpy.prod(shape))))
            self.assertEqual(emb.shape, (shape or (embedding_dim,)))
            self.assertEqual(emb.num_embeddings, self.num)

            # check attributes
            self.assertIs(emb.initializer, initializer)
            self.assertIs(emb.normalizer, normalizer)
            self.assertIs(emb.constrainer, constrainer)
            self.assertIs(emb.regularizer, regularizer)


class KullbackLeiblerTests(unittest.TestCase):
    """Tests for the vectorized computation of KL divergences."""

    batch_size: int = 2
    num_heads: int = 3
    num_relations: int = 5
    num_tails: int = 7
    d: int = 11

    def setUp(self) -> None:  # noqa: D102
        dims = dict(h=self.num_heads, r=self.num_relations, t=self.num_tails)
        (self.h_mean, self.r_mean, self.t_mean), (self.h_var, self.r_var, self.t_var) = [
            [
                convert_to_canonical_shape(
                    x=torch.rand(self.batch_size, num, self.d),
                    dim=dim,
                    num=num,
                    batch_size=self.batch_size,
                )
                for dim, num in dims.items()
            ]
            for _ in ("mean", "diagonal_covariance")
        ]
        # ensure positivity
        self.h_var, self.r_var, self.t_var = [x.exp() for x in (self.h_var, self.r_var, self.t_var)]

    def _get(self, name: str):
        if name == "h":
            mean, var = self.h_mean, self.h_var
        elif name == "r":
            mean, var = self.r_mean, self.r_var
        elif name == "t":
            mean, var = self.t_mean, self.t_var
        elif name == "e":
            mean, var = self.h_mean - self.t_mean, self.h_var + self.t_var
        else:
            raise ValueError
        return GaussianDistribution(mean=mean, diagonal_covariance=var)

    def _get_kl_similarity_torch(self):
        # compute using pytorch
        e_mean = self.h_mean - self.t_mean
        e_var = self.h_var + self.t_var
        r_mean, r_var = self.r_var, self.r_mean
        self.assertTrue((e_var > 0).all())
        sim2 = torch.empty(self.batch_size, self.num_heads, self.num_relations, self.num_tails)
        for bi, hi, ri, ti in itertools.product(
            range(self.batch_size),
            range(self.num_heads),
            range(self.num_relations),
            range(self.num_tails),
        ):
            # prepare distributions
            e_loc = e_mean[bi, hi, 0, ti, :]
            r_loc = r_mean[bi, 0, ri, 0, :]
            e_cov = torch.diag(e_var[bi, hi, 0, ti, :])
            r_cov = torch.diag(r_var[bi, 0, ri, 0, :])
            p = torch.distributions.MultivariateNormal(
                loc=e_loc,
                covariance_matrix=e_cov,
            )
            q = torch.distributions.MultivariateNormal(
                loc=r_loc,
                covariance_matrix=r_cov,
            )
            sim2[bi, hi, ri, ti] = -torch.distributions.kl_divergence(p=p, q=q).view(-1)
        return sim2

    def test_against_torch_builtin(self):
        """Compare value against torch.distributions."""
        # compute using pykeen
        h, r, t = [self._get(name=name) for name in "hrt"]
        sim = kullback_leibler_similarity(h=h, r=r, t=t, exact=True)
        sim2 = _torch_kl_similarity(h=h, r=r, t=t)
        self.assertTrue(torch.allclose(sim, sim2), msg=f'Difference: {(sim - sim2).abs()}')

    def test_self_similarity(self):
        """Check value of similarity to self."""
        # e: (batch_size, num_heads, num_tails, d)
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties
        # divergence = 0 => similarity = -divergence = 0
        # (h - t), r
        r = self._get(name="r")
        h = GaussianDistribution(mean=2 * r.mean, diagonal_covariance=0.5 * r.diagonal_covariance)
        t = GaussianDistribution(mean=r.mean, diagonal_covariance=0.5 * r.diagonal_covariance)
        sim = kullback_leibler_similarity(h=h, r=r, t=t, exact=True)
        self.assertTrue(torch.allclose(sim, torch.zeros_like(sim)), msg=f'Sim: {sim}')

    def test_value_range(self):
        """Check the value range."""
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties
        # divergence >= 0 => similarity = -divergence <= 0
        h, r, t = [self._get(name=name) for name in "hrt"]
        sim = kullback_leibler_similarity(h=h, r=r, t=t, exact=True)
        self.assertTrue((sim <= 0).all())
