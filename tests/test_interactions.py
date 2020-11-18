# -*- coding: utf-8 -*-

"""Tests for interaction functions."""

import unittest
from operator import itemgetter
from typing import Any, Callable, Collection, Generic, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union
from unittest.case import SkipTest

import numpy
import torch

import pykeen.nn.modules
from pykeen.nn import functional as pkf
from pykeen.nn.modules import Interaction, StatelessInteraction, TranslationalInteraction
from pykeen.typing import Representation
from pykeen.utils import get_subclasses

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

    functional_form: Callable[..., torch.FloatTensor]

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

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:
        return dict(h=h, r=r, t=t)

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
                kwargs = self._prepare_functional_input(h, r, t)
                scores_f = self.__class__.functional_form(**kwargs)
                assert torch.allclose(scores, scores_f)

    def test_scores(self):
        """Test individual scores."""
        self.instance.eval()
        for _ in range(10):
            h, r, t = self._get_hrt((1, 1), (1, 1), (1, 1))
            kwargs = self._prepare_functional_input(h, r, t)

            # calculate by functional
            scores_f = self.__class__.functional_form(**kwargs).item()

            # calculate manually
            scores_f_manual = self._exp_score(**kwargs).item()
            assert scores_f_manual == scores_f

    def _exp_score(self, **kwargs) -> torch.FloatTensor:
        """Compute the expected score for a single-score batch."""
        raise SkipTest("No score check implemented.")


class ComplExTests(InteractionTests, unittest.TestCase):
    """Tests for ComplEx interaction function."""

    cls = pykeen.nn.modules.ComplExInteraction
    functional_form = pkf.complex_interaction


def _get_key_sorted_kwargs_values(kwargs: Mapping[str, Any]) -> Sequence[Any]:
    return list(map(itemgetter(1), sorted(kwargs.items(), key=itemgetter(0))))


class ConvETests(InteractionTests, unittest.TestCase):
    """Tests for ConvE interaction function."""

    cls = pykeen.nn.modules.ConvEInteraction
    functional_form = pkf.conve_interaction
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

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        return dict(
            h=h,
            r=r,
            t=t[0],
            t_bias=t[1],
            input_channels=self.instance.input_channels,
            embedding_height=self.instance.embedding_height,
            embedding_width=self.instance.embedding_width,
            hr2d=self.instance.hr2d,
            hr1d=self.instance.hr1d,
        )

    def _exp_score(self, **kwargs) -> torch.FloatTensor:
        height, width, h, hr1d, hr2d, input_channels, r, t, t_bias = _get_key_sorted_kwargs_values(kwargs)
        x = torch.cat([
            h.view(1, input_channels, height, width),
            r.view(1, input_channels, height, width)
        ], dim=2)
        x = hr2d(x)
        x = x.view(-1, numpy.prod(x.shape[-3:]))
        x = hr1d(x)
        return (x.view(1, -1) * t.view(1, -1)).sum() + t_bias


class ConvKBTests(InteractionTests, unittest.TestCase):
    """Tests for ConvKB interaction function."""

    cls = pykeen.nn.modules.ConvKBInteraction
    functional_form = pkf.convkb_interaction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
        num_filters=2 * InteractionTests.dim - 1,
    )

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        return dict(
            h=h,
            r=r,
            t=t,
            conv=self.instance.conv,
            activation=self.instance.activation,
            hidden_dropout=self.instance.hidden_dropout,
            linear=self.instance.linear,
        )


class DistMultTests(InteractionTests, unittest.TestCase):
    """Tests for DistMult interaction function."""

    cls = pykeen.nn.modules.DistMultInteraction
    functional_form = pkf.distmult_interaction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        return (h * r * t).sum(dim=-1)


class ERMLPTests(InteractionTests, unittest.TestCase):
    """Tests for ERMLP interaction function."""

    cls = pykeen.nn.modules.ERMLPInteraction
    functional_form = pkf.ermlp_interaction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
        hidden_dim=2 * InteractionTests.dim - 1,
    )

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        return dict(
            h=h,
            r=r,
            t=t,
            hidden=self.instance.hidden,
            activation=self.instance.activation,
            final=self.instance.hidden_to_score,
        )


class ERMLPETests(InteractionTests, unittest.TestCase):
    """Tests for ERMLP-E interaction function."""

    cls = pykeen.nn.modules.ERMLPEInteraction
    functional_form = pkf.ermlpe_interaction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
        hidden_dim=2 * InteractionTests.dim - 1,
    )

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        return dict(h=h, r=r, t=t, mlp=self.instance.mlp)


class HolETests(InteractionTests, unittest.TestCase):
    """Tests for HolE interaction function."""

    cls = pykeen.nn.modules.HolEInteraction
    functional_form = pkf.hole_interaction


class NTNTests(InteractionTests, unittest.TestCase):
    """Tests for NTN interaction function."""

    cls = pykeen.nn.modules.NTNInteraction
    functional_form = pkf.ntn_interaction

    num_slices: int = 11
    shape_kwargs = dict(
        k=11,
    )

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        return dict(h=h, t=t, w=r[0], b=r[1], u=r[2], vh=r[3], vt=r[4], activation=self.instance.non_linearity)

    def _exp_score(self, **kwargs) -> torch.FloatTensor:
        # f(h,r,t) = u_r^T act(h W_r t + V_r h + V_r' t + b_r)
        # shapes:
        # w: (k, dim, dim)
        # vh/vt: (k, dim)
        # b/u: (k,)
        activation, b, h, t, u, vh, vt, w = _get_key_sorted_kwargs_values(kwargs)
        h, t = h.view(1, self.dim, 1), t.view(1, self.dim, 1)
        w = w.view(self.num_slices, self.dim, self.dim)
        vh, vt = [v.view(self.num_slices, 1, self.dim) for v in (vh, vt)]
        b = b.view(self.num_slices, 1, 1)
        u = u.view(self.num_slices, )
        x = activation(h.transpose(-2, -1) @ w @ t + vh @ h + vt @ t + b).view(self.num_slices)
        return (x * u).sum()


class ProjETests(InteractionTests, unittest.TestCase):
    """Tests for ProjE interaction function."""

    cls = pykeen.nn.modules.ProjEInteraction
    functional_form = pkf.proje_interaction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
    )

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        return dict(
            h=h,
            r=r,
            t=t,
            d_e=self.instance.d_e,
            d_r=self.instance.d_r,
            b_c=self.instance.b_c,
            b_p=self.instance.b_p,
            activation=self.instance.inner_non_linearity,
        )


class RESCALTests(InteractionTests, unittest.TestCase):
    """Tests for RESCAL interaction function."""

    cls = pykeen.nn.modules.RESCALInteraction
    functional_form = pkf.rescal_interaction


class KG2ETests(InteractionTests, unittest.TestCase):
    """Tests for KG2E interaction function."""

    cls = pykeen.nn.modules.KG2EInteraction
    functional_form = pkf.kg2e_interaction

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        return dict(h_mean=h[0], h_var=h[1], r_mean=r[0], r_var=r[1], t_mean=t[0], t_var=t[1])


class TuckerTests(InteractionTests, unittest.TestCase):
    """Tests for Tucker interaction function."""

    cls = pykeen.nn.modules.TuckerInteraction
    functional_form = pkf.tucker_interaction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
    )


class RotatETests(InteractionTests, unittest.TestCase):
    """Tests for RotatE interaction function."""

    cls = pykeen.nn.modules.RotatEInteraction
    functional_form = pkf.rotate_interaction


class TranslationalInteractionTests(InteractionTests):
    """Common tests for translational interaction."""

    kwargs = dict(
        p=2,
    )

    def _additional_score_checks(self, scores):
        assert (scores <= 0).all()

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        return dict(
            h=h,
            r=r,
            t=t,
            p=self.instance.p,
            power_norm=self.instance.power_norm,
        )


class TransDTests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for TransD interaction function."""

    cls = pykeen.nn.modules.TransDInteraction
    functional_form = pkf.transd_interaction
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

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        kwargs = dict(super()._prepare_functional_input(h=h, r=r, t=t))
        kwargs.update(h=h[0], r=r[0], t=t[0], h_p=h[1], r_p=r[1], t_p=t[1])
        return kwargs


class TransETests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for TransE interaction function."""

    cls = pykeen.nn.modules.TransEInteraction
    functional_form = pkf.transe_interaction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        return -(h + r - t).norm(p=2, dim=-1)


class TransHTests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for TransH interaction function."""

    cls = pykeen.nn.modules.TransHInteraction
    functional_form = pkf.transh_interaction

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        kwargs = dict(super()._prepare_functional_input(h=h, r=r, t=t))
        w_r, d_r = kwargs.pop("r")
        kwargs.update(w_r=w_r, d_r=d_r)
        return kwargs


class TransRTests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for TransR interaction function."""

    cls = pykeen.nn.modules.TransRInteraction
    functional_form = pkf.transr_interaction
    shape_kwargs = dict(
        e=3,
    )

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        kwargs = dict(super()._prepare_functional_input(h=h, r=r, t=t))
        r, m_r = kwargs.pop("r")
        kwargs.update(r=r, m_r=m_r)
        return kwargs

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
    functional_form = pkf.structured_embedding_interaction

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        kwargs = dict(super()._prepare_functional_input(h=h, r=r, t=t))
        r_h, r_t = kwargs.pop("r")
        kwargs.update(r_h=r_h, r_t=r_t)
        return kwargs


class UMTests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for UM interaction function."""

    cls = pykeen.nn.modules.UnstructuredModelInteraction
    functional_form = pkf.unstructured_model_interaction

    def _prepare_functional_input(
        self,
        h: Union[Representation, Sequence[Representation]],
        r: Union[Representation, Sequence[Representation]],
        t: Union[Representation, Sequence[Representation]],
    ) -> Mapping[str, Any]:  # noqa: D102
        kwargs = dict(super()._prepare_functional_input(h=h, r=r, t=t))
        kwargs.pop("r")
        return kwargs


class InteractionTestsTest(TestsTest[Interaction], unittest.TestCase):
    """Test for tests for all interaction functions."""

    base_cls = Interaction
    base_test = InteractionTests
    skip_cls = {
        TranslationalInteraction,
        StatelessInteraction,
    }
