# -*- coding: utf-8 -*-

"""Tests for interaction functions."""

import unittest
from typing import Any, Collection, Generic, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union
from unittest.case import SkipTest

import numpy
import torch

import pykeen.nn.modules
from pykeen.nn.modules import Interaction, TranslationalInteraction
from pykeen.typing import Representation
from pykeen.utils import get_subclasses, project_entity, view_complex

T = TypeVar("T")


class GenericTests(Generic[T]):
    """Generic tests."""

    cls: Type[T]
    kwargs: Optional[Mapping[str, Any]] = None
    instance: T

    def setUp(self) -> None:
        """Set up the generic testing method."""
        kwargs = self.kwargs or {}
        kwargs = self._pre_instantiation_hook(kwargs=dict(kwargs))
        self.instance = self.cls(**kwargs)
        self.post_instantiation_hook()

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Perform actions before instantiation, potentially modyfing kwargs."""
        return kwargs

    def post_instantiation_hook(self) -> None:
        """Perform actions after instantiation."""


class TestsTest(Generic[T]):
    """A generic test for tests."""

    base_cls: Type[T]
    base_test: Type[GenericTests[T]]
    skip_cls: Collection[T] = tuple()

    def test_testing(self):
        """Check that there is a test for all subclasses."""
        to_test = set(get_subclasses(self.base_cls)).difference(self.skip_cls)
        tested = (test_cls.cls for test_cls in get_subclasses(self.base_test) if hasattr(test_cls, "cls"))
        not_tested = to_test.difference(tested)
        assert not not_tested, not_tested


class InteractionTests(GenericTests[pykeen.nn.modules.Interaction]):
    """Generic test for interaction functions."""

    dim: int = 2
    batch_size: int = 3
    num_relations: int = 5
    num_entities: int = 7

    shape_kwargs = dict()

    def _get_hrt(
        self,
        *shapes: Tuple[int, ...],
    ) -> Tuple[Union[Representation, Sequence[Representation]], ...]:
        self.shape_kwargs.setdefault("d", self.dim)
        result = tuple(
            tuple(
                torch.rand(*prefix_shape, *(self.shape_kwargs[dim] for dim in weight_shape), requires_grad=True)
                for weight_shape in weight_shapes
            )
            for prefix_shape, weight_shapes in zip(
                shapes,
                [self.cls.entity_shape, self.cls.relation_shape, self.cls.entity_shape],
            )
        )
        return tuple(pykeen.nn.modules._unpack_singletons(*result))

    def _check_scores(self, scores: torch.FloatTensor, exp_shape: Tuple[int, ...]):
        """Check shape, dtype and gradients of scores."""
        assert torch.is_tensor(scores)
        assert scores.dtype == torch.float32
        assert scores.ndimension() == len(exp_shape)
        assert scores.shape == exp_shape
        assert scores.requires_grad
        self._additional_score_checks(scores)

    def _additional_score_checks(self, scores):
        """Additional checks for scores."""

    def test_score_hrt(self):
        """Test score_hrt."""
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.batch_size,),
            (self.batch_size,),
        )
        scores = self.instance.score_hrt(h=h, r=r, t=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, 1))

    def test_score_h(self):
        """Test score_h."""
        h, r, t = self._get_hrt(
            (self.num_entities,),
            (self.batch_size,),
            (self.batch_size,),
        )
        scores = self.instance.score_h(all_entities=h, r=r, t=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, self.num_entities))

    def test_score_h_slicing(self):
        """Test score_h with slicing."""
        #: The equivalence for models with batch norm only holds in evaluation mode
        self.instance.eval()
        h, r, t = self._get_hrt(
            (self.num_entities,),
            (self.batch_size,),
            (self.batch_size,),
        )
        scores = self.instance.score_h(all_entities=h, r=r, t=t, slice_size=self.num_entities // 2 + 1)
        scores_no_slice = self.instance.score_h(all_entities=h, r=r, t=t, slice_size=None)
        assert torch.allclose(scores, scores_no_slice), f'Differences: {scores - scores_no_slice}'

    def test_score_r(self):
        """Test score_r."""
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.num_relations,),
            (self.batch_size,),
        )
        scores = self.instance.score_r(h=h, all_relations=r, t=t)
        if len(self.cls.relation_shape) == 0:
            exp_shape = (self.batch_size, 1)
        else:
            exp_shape = (self.batch_size, self.num_relations)
        self._check_scores(scores=scores, exp_shape=exp_shape)

    def test_score_r_slicing(self):
        """Test score_r with slicing."""
        if len(self.cls.relation_shape) == 0:
            raise SkipTest("No use in slicing relations for models without relation information.")
        #: The equivalence for models with batch norm only holds in evaluation mode
        self.instance.eval()
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.num_relations,),
            (self.batch_size,),
        )
        scores = self.instance.score_r(h=h, all_relations=r, t=t, slice_size=self.num_relations // 2 + 1)
        scores_no_slice = self.instance.score_r(h=h, all_relations=r, t=t, slice_size=None)
        assert torch.allclose(scores, scores_no_slice), f'Differences: {scores - scores_no_slice}'

    def test_score_t(self):
        """Test score_t."""
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.batch_size,),
            (self.num_entities,),
        )
        scores = self.instance.score_t(h=h, r=r, all_entities=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, self.num_entities))

    def test_score_t_slicing(self):
        """Test score_t with slicing."""
        #: The equivalence for models with batch norm only holds in evaluation mode
        self.instance.eval()
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.batch_size,),
            (self.num_entities,),
        )
        scores = self.instance.score_t(h=h, r=r, all_entities=t, slice_size=self.num_entities // 2 + 1)
        scores_no_slice = self.instance.score_t(h=h, r=r, all_entities=t, slice_size=None)
        assert torch.allclose(scores, scores_no_slice), f'Differences: {scores - scores_no_slice}'

    def _get_test_shapes(self) -> Collection[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """Return a set of test shapes for (h, r, t)."""
        return (
            (  # single score
                (1, 1),
                (1, 1),
                (1, 1),
            ),
            (  # score_r with multi-t
                (self.batch_size, 1),
                (1, self.num_relations),
                (self.batch_size, self.num_entities // 2 + 1),
            ),
            (  # score_r with multi-t and broadcasted head
                (1, 1),
                (1, self.num_relations),
                (self.batch_size, self.num_entities),
            ),
            (  # full cwa
                (1, self.num_entities),
                (1, self.num_relations),
                (1, self.num_entities),
            ),
        )

    def _get_output_shape(
        self,
        hs: Tuple[int, int],
        rs: Tuple[int, int],
        ts: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        batch_size = max(hs[0], rs[0], ts[0])
        nh, nr, nt = hs[1], rs[1], ts[1]
        if len(self.cls.relation_shape) == 0:
            nr = 1
        if len(self.cls.entity_shape) == 0:
            nh = nt = 1
        return batch_size, nh, nr, nt

    def test_forward(self):
        """Test forward."""
        for hs, rs, ts in self._get_test_shapes():
            if any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)) for m in self.instance.modules()):
                # TODO: do we need to skip this for every combination? or only if batch_size = 1?
                continue
            with self.subTest(f"forward({hs}, {rs}, {ts})"):
                h, r, t = self._get_hrt(hs, rs, ts)
                scores = self.instance(h=h, r=r, t=t)
                expected_shape = self._get_output_shape(hs, rs, ts)
                self._check_scores(scores=scores, exp_shape=expected_shape)
            with self.subTest(f"forward({hs}, {rs}, {ts}) - consistency with functional"):
                kwargs = self.instance._prepare_for_functional(h=h, r=r, t=t)
                scores_f = self.cls.func(**kwargs)
                assert torch.allclose(scores, scores_f)

    def test_scores(self):
        """Test individual scores."""
        self.instance.eval()
        for _ in range(10):
            h, r, t = self._get_hrt((1, 1), (1, 1), (1, 1))
            kwargs = self.instance._prepare_for_functional(h=h, r=r, t=t)

            # calculate by functional
            scores_f = self.cls.func(**kwargs)

            # calculate manually
            scores_f_manual = self._exp_score(**kwargs)
            assert torch.allclose(scores_f_manual, scores_f)

    def _exp_score(self, **kwargs) -> torch.FloatTensor:
        """Compute the expected score for a single-score batch."""
        raise SkipTest("No score check implemented.")


class ComplExTests(InteractionTests, unittest.TestCase):
    """Tests for ComplEx interaction function."""

    cls = pykeen.nn.modules.ComplExInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        h, r, t = [view_complex(x) for x in (h, r, t)]
        return (h * r * t).sum().real


class ConvETests(InteractionTests, unittest.TestCase):
    """Tests for ConvE interaction function."""

    cls = pykeen.nn.modules.ConvEInteraction
    kwargs = dict(
        embedding_height=1,
        embedding_width=2,
        kernel_height=2,
        kernel_width=1,
        embedding_dim=InteractionTests.dim,
    )

    def _get_hrt(
        self,
        *shapes: Tuple[int, ...],
        **kwargs,
    ) -> Tuple[Union[Representation, Sequence[Representation]], ...]:  # noqa: D102
        h, r, t = super()._get_hrt(*shapes, **kwargs)
        t_bias = torch.rand_like(t[..., 0, None])
        return h, r, (t, t_bias)

    def _exp_score(
        self, embedding_height, embedding_width, h, hr1d, hr2d, input_channels, r, t, t_bias
    ) -> torch.FloatTensor:
        x = torch.cat([
            h.view(1, input_channels, embedding_height, embedding_width),
            r.view(1, input_channels, embedding_height, embedding_width)
        ], dim=2)
        x = hr2d(x)
        x = x.view(-1, numpy.prod(x.shape[-3:]))
        x = hr1d(x)
        return (x.view(1, -1) * t.view(1, -1)).sum() + t_bias


class ConvKBTests(InteractionTests, unittest.TestCase):
    """Tests for ConvKB interaction function."""

    cls = pykeen.nn.modules.ConvKBInteraction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
        num_filters=2 * InteractionTests.dim - 1,
    )

    def _exp_score(self, h, r, t, conv, activation, hidden_dropout, linear) -> torch.FloatTensor:  # noqa: D102
        # W_L drop(act(W_C \ast ([h; r; t]) + b_C)) + b_L
        x = torch.stack([x.view(-1) for x in (h, r, t)], dim=1).view(1, 1, -1, 3)
        return linear(hidden_dropout(activation(conv(x).view(1, -1))))


class DistMultTests(InteractionTests, unittest.TestCase):
    """Tests for DistMult interaction function."""

    cls = pykeen.nn.modules.DistMultInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        return (h * r * t).sum(dim=-1)


class ERMLPTests(InteractionTests, unittest.TestCase):
    """Tests for ERMLP interaction function."""

    cls = pykeen.nn.modules.ERMLPInteraction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
        hidden_dim=2 * InteractionTests.dim - 1,
    )

    def _exp_score(self, h, r, t, hidden, activation, final) -> torch.FloatTensor:
        x = torch.cat([x.view(-1) for x in (h, r, t)])
        return final(activation(hidden(x)))


class ERMLPETests(InteractionTests, unittest.TestCase):
    """Tests for ERMLP-E interaction function."""

    cls = pykeen.nn.modules.ERMLPEInteraction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
        hidden_dim=2 * InteractionTests.dim - 1,
    )

    def _exp_score(self, h, r, t, mlp) -> torch.FloatTensor:  # noqa: D102
        x = torch.cat([x.view(1, -1) for x in (h, r)], dim=-1)
        return mlp(x).view(1, -1) @ t.view(-1, 1)


class HolETests(InteractionTests, unittest.TestCase):
    """Tests for HolE interaction function."""

    cls = pykeen.nn.modules.HolEInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        h, t = [torch.fft.rfft(x.view(1, -1), dim=-1) for x in (h, t)]
        h = torch.conj(h)
        c = torch.fft.irfft(h * t, n=h.shape[-1], dim=-1)
        return (c * r).sum()


def _strip_dim(*x):
    return [xx.view(xx.shape[2:]) for xx in x]


class NTNTests(InteractionTests, unittest.TestCase):
    """Tests for NTN interaction function."""

    cls = pykeen.nn.modules.NTNInteraction

    num_slices: int = 11
    shape_kwargs = dict(
        k=11,
    )

    def _exp_score(self, h, t, w, vt, vh, b, u, activation) -> torch.FloatTensor:
        # f(h,r,t) = u_r^T act(h W_r t + V_r h + V_r' t + b_r)
        # shapes: w: (k, dim, dim), vh/vt: (k, dim), b/u: (k,), h/t: (dim,)
        # remove batch/num dimension
        h, t, w, vt, vh, b, u = _strip_dim(h, t, w, vt, vh, b, u)
        score = 0.
        for i in range(u.shape[-1]):
            score = score + u[i] * activation(
                h.view(1, self.dim) @ w[i] @ t.view(self.dim, 1) +
                (vh[i] * h.view(-1)).sum() +
                (vt[i] * t.view(-1)).sum() +
                b[i]
            )
        return score


class ProjETests(InteractionTests, unittest.TestCase):
    """Tests for ProjE interaction function."""

    cls = pykeen.nn.modules.ProjEInteraction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
    )

    def _exp_score(self, h, r, t, d_e, d_r, b_c, b_p, activation) -> torch.FloatTensor:
        # f(h, r, t) = g(t z(D_e h + D_r r + b_c) + b_p)
        h, r, t = _strip_dim(h, r, t)
        return (t * activation((d_e * h) + (d_r * r) + b_c)).sum() + b_p


class RESCALTests(InteractionTests, unittest.TestCase):
    """Tests for RESCAL interaction function."""

    cls = pykeen.nn.modules.RESCALInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        # f(h, r, t) = h @ r @ t
        h, r, t = _strip_dim(h, r, t)
        return h.view(1, -1) @ r @ t.view(-1, 1)


class KG2ETests(InteractionTests, unittest.TestCase):
    """Tests for KG2E interaction function."""

    cls = pykeen.nn.modules.KG2EInteraction


class TuckerTests(InteractionTests, unittest.TestCase):
    """Tests for Tucker interaction function."""

    cls = pykeen.nn.modules.TuckerInteraction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
    )


class RotatETests(InteractionTests, unittest.TestCase):
    """Tests for RotatE interaction function."""

    cls = pykeen.nn.modules.RotatEInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        h, r, t = _strip_dim(*(view_complex(x) for x in (h, r, t)))
        hr = h * r
        d = hr - t
        return -(d.abs() ** 2).sum().sqrt()


class TranslationalInteractionTests(InteractionTests):
    """Common tests for translational interaction."""

    kwargs = dict(
        p=2,
    )

    def _additional_score_checks(self, scores):
        assert (scores <= 0).all()


class TransDTests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for TransD interaction function."""

    cls = pykeen.nn.modules.TransDInteraction
    shape_kwargs = dict(
        e=3,
    )

    def test_manual_small_relation_dim(self):
        """Manually test the value of the interaction function."""
        # entity embeddings
        h = t = torch.as_tensor(data=[2., 2.], dtype=torch.float).view(1, 2)
        h_p = t_p = torch.as_tensor(data=[3., 3.], dtype=torch.float).view(1, 2)

        # relation embeddings
        r = torch.as_tensor(data=[4.], dtype=torch.float).view(1, 1)
        r_p = torch.as_tensor(data=[5.], dtype=torch.float).view(1, 1)

        # Compute Scores
        scores = self.instance.score_hrt(h=(h, h_p), r=(r, r_p), t=(t, t_p))
        first_score = scores[0].item()
        self.assertAlmostEqual(first_score, -16, delta=0.01)

    def test_manual_big_relation_dim(self):
        """Manually test the value of the interaction function."""
        # entity embeddings
        h = t = torch.as_tensor(data=[2., 2.], dtype=torch.float).view(1, 2)
        h_p = t_p = torch.as_tensor(data=[3., 3.], dtype=torch.float).view(1, 2)

        # relation embeddings
        r = torch.as_tensor(data=[3., 3., 3.], dtype=torch.float).view(1, 3)
        r_p = torch.as_tensor(data=[4., 4., 4.], dtype=torch.float).view(1, 3)

        # Compute Scores
        scores = self.instance.score_hrt(h=(h, h_p), r=(r, r_p), t=(t, t_p))
        self.assertAlmostEqual(scores.item(), -27, delta=0.01)

    def _exp_score(self, h, r, t, h_p, r_p, t_p, p, power_norm) -> torch.FloatTensor:  # noqa: D102
        assert power_norm
        h_bot = project_entity(e=h, e_p=h_p, r_p=r_p)
        t_bot = project_entity(e=t, e_p=t_p, r_p=r_p)
        return -((h_bot + r - t_bot) ** p).sum()


class TransETests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for TransE interaction function."""

    cls = pykeen.nn.modules.TransEInteraction

    def _exp_score(self, h, r, t, p, power_norm) -> torch.FloatTensor:
        assert not power_norm
        return -(h + r - t).norm(p=p, dim=-1)


class TransHTests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for TransH interaction function."""

    cls = pykeen.nn.modules.TransHInteraction

    def _exp_score(self, h, w_r, d_r, t, p, power_norm) -> torch.FloatTensor:  # noqa: D102
        assert not power_norm
        h, w_r, d_r, t = _strip_dim(h, w_r, d_r, t)
        h, t = [x - (x * w_r).sum() * w_r for x in (h, t)]
        return -(h + d_r - t).norm(p=p)


class TransRTests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for TransR interaction function."""

    cls = pykeen.nn.modules.TransRInteraction
    shape_kwargs = dict(
        e=3,
    )

    def test_manual(self):
        """Manually test the value of the interaction function."""
        # Compute Scores
        h = torch.as_tensor(data=[2, 2], dtype=torch.float32).view(1, 2)
        r = torch.as_tensor(data=[4, 4], dtype=torch.float32).view(1, 2)
        m_r = torch.as_tensor(data=[5, 5, 6, 6], dtype=torch.float32).view(1, 2, 2)
        t = torch.as_tensor(data=[2, 2], dtype=torch.float32).view(1, 2)
        scores = self.instance.score_hrt(h=h, r=(r, m_r), t=t)
        first_score = scores[0].item()
        self.assertAlmostEqual(first_score, -32, delta=1.0e-04)


class SETests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for SE interaction function."""

    cls = pykeen.nn.modules.StructuredEmbeddingInteraction


class UMTests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for UM interaction function."""

    cls = pykeen.nn.modules.UnstructuredModelInteraction


class InteractionTestsTest(TestsTest[Interaction], unittest.TestCase):
    """Test for tests for all interaction functions."""

    base_cls = Interaction
    base_test = InteractionTests
    skip_cls = {
        TranslationalInteraction,
    }
