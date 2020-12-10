# -*- coding: utf-8 -*-

"""Tests for interaction functions."""

import logging
import unittest
from abc import abstractmethod
from typing import Collection, Sequence, Tuple, Union
from unittest.case import SkipTest

import numpy
import torch

import pykeen.nn.modules
from pykeen.models.multimodal.base import LiteralInteraction
from pykeen.nn.functional import distmult_interaction
from pykeen.nn.modules import Interaction, TranslationalInteraction
from pykeen.testing import base as ptb
from pykeen.typing import Representation
from pykeen.utils import clamp_norm, project_entity, strip_dim, view_complex

logger = logging.getLogger(__name__)


class InteractionTests(ptb.GenericTests[pykeen.nn.modules.Interaction]):
    """Generic test for interaction functions."""

    dim: int = 2
    batch_size: int = 3
    num_relations: int = 5
    num_entities: int = 7

    shape_kwargs = dict()

    def post_instantiation_hook(self) -> None:
        """Initialize parameters."""
        self.instance.reset_parameters()

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
        self._check_close_scores(scores=scores, scores_no_slice=scores_no_slice)

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
        self._check_close_scores(scores=scores, scores_no_slice=scores_no_slice)

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
        self._check_close_scores(scores=scores, scores_no_slice=scores_no_slice)

    def _check_close_scores(self, scores, scores_no_slice):
        self.assertTrue(torch.isfinite(scores).all(), msg=f'Normal scores had nan:\n\t{scores}')
        self.assertTrue(torch.isfinite(scores_no_slice).all(), msg=f'Slice scores had nan\n\t{scores}')
        self.assertTrue(torch.allclose(scores, scores_no_slice), msg=f'Differences: {scores - scores_no_slice}')

    def _get_test_shapes(self) -> Collection[Tuple[
        Tuple[int, int, int, int],
        Tuple[int, int, int, int],
        Tuple[int, int, int, int],
    ]]:
        """Return a set of test shapes for (h, r, t)."""
        return (
            (  # single score
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (1, 1, 1, 1),
            ),
            (  # score_r with multi-t
                (self.batch_size, 1, 1, 1),
                (1, 1, self.num_relations, 1),
                (self.batch_size, 1, 1, self.num_entities // 2 + 1),
            ),
            (  # score_r with multi-t and broadcasted head
                (1, 1, 1, 1),
                (1, 1, self.num_relations, 1),
                (self.batch_size, 1, 1, self.num_entities),
            ),
            (  # full cwa
                (1, self.num_entities, 1, 1),
                (1, 1, self.num_relations, 1),
                (1, 1, 1, self.num_entities),
            ),
        )

    def _get_output_shape(
        self,
        hs: Tuple[int, int, int, int],
        rs: Tuple[int, int, int, int],
        ts: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        result = [max(ds) for ds in zip(hs, rs, ts)]
        if len(self.instance.entity_shape) == 0:
            result[1] = result[3] = 1
        if len(self.instance.relation_shape) == 0:
            result[2] = 1
        return tuple(result)

    def test_forward(self):
        """Test forward."""
        for hs, rs, ts in self._get_test_shapes():
            try:
                h, r, t = self._get_hrt(hs, rs, ts)
                scores = self.instance(h=h, r=r, t=t)
                expected_shape = self._get_output_shape(hs, rs, ts)
                self._check_scores(scores=scores, exp_shape=expected_shape)
            except ValueError as error:
                # check whether the error originates from batch norm for single element batches
                small_batch_size = any(s[0] == 1 for s in (hs, rs, ts))
                has_batch_norm = any(
                    isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
                    for m in self.instance.modules()
                )
                if small_batch_size and has_batch_norm:
                    logger.warning(
                        f"Skipping test for shapes {hs}, {rs}, {ts} because too small batch size for batch norm",
                    )
                    continue
                raise error

    def test_forward_consistency_with_functional(self):
        """Test forward's consistency with functional."""
        # set in eval mode (otherwise there are non-deterministic factors like Dropout
        self.instance.eval()
        for hs, rs, ts in self._get_test_shapes():
            h, r, t = self._get_hrt(hs, rs, ts)
            scores = self.instance(h=h, r=r, t=t)
            kwargs = self.instance._prepare_for_functional(h=h, r=r, t=t)
            scores_f = self.cls.func(**kwargs)
            assert torch.allclose(scores, scores_f)

    def test_scores(self):
        """Test individual scores."""
        # set in eval mode (otherwise there are non-deterministic factors like Dropout
        self.instance.eval()
        for _ in range(10):
            # test multiple different initializations
            self.instance.reset_parameters()
            h, r, t = self._get_hrt((1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
            kwargs = self.instance._prepare_for_functional(h=h, r=r, t=t)

            # calculate by functional
            scores_f = self.cls.func(**kwargs).view(-1)

            # calculate manually
            scores_f_manual = self._exp_score(**kwargs).view(-1)
            assert torch.allclose(scores_f_manual, scores_f), f'Diff: {scores_f_manual - scores_f}'

    @abstractmethod
    def _exp_score(self, **kwargs) -> torch.FloatTensor:
        """Compute the expected score for a single-score batch."""
        raise NotImplementedError(f"{self.cls.__name__}({sorted(kwargs.keys())})")


class ComplExTests(InteractionTests, unittest.TestCase):
    """Tests for ComplEx interaction function."""

    cls = pykeen.nn.modules.ComplExInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        h, r, t = [view_complex(x) for x in (h, r, t)]
        return (h * r * torch.conj(t)).sum().real


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
        self, embedding_height, embedding_width, h, hr1d, hr2d, input_channels, r, t, t_bias,
    ) -> torch.FloatTensor:
        x = torch.cat([
            h.view(1, input_channels, embedding_height, embedding_width),
            r.view(1, input_channels, embedding_height, embedding_width),
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
        # prepare conv input (N, C, H, W)
        x = torch.stack([x.view(-1) for x in (h, r, t)], dim=1).view(1, 1, -1, 3)
        x = conv(x)
        x = hidden_dropout(activation(x))
        return linear(x.view(1, -1))


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


class NTNTests(InteractionTests, unittest.TestCase):
    """Tests for NTN interaction function."""

    cls = pykeen.nn.modules.NTNInteraction

    num_slices: int = 11
    shape_kwargs = dict(
        k=11,
    )

    def _exp_score(self, h, t, w, vt, vh, b, u, activation) -> torch.FloatTensor:
        # f(h,r,t) = u_r^T act(h W_r t + V_r h + V_r t + b_r)
        # shapes: w: (k, dim, dim), vh/vt: (k, dim), b/u: (k,), h/t: (dim,)
        # remove batch/num dimension
        h, t, w, vt, vh, b, u = strip_dim(h, t, w, vt, vh, b, u)
        score = 0.
        for i in range(u.shape[-1]):
            first_part = h.view(1, self.dim) @ w[i] @ t.view(self.dim, 1)
            second_part = (vh[i] * h.view(-1)).sum()
            third_part = (vt[i] * t.view(-1)).sum()
            score = score + u[i] * activation(first_part + second_part + third_part + b[i])
        return score


class ProjETests(InteractionTests, unittest.TestCase):
    """Tests for ProjE interaction function."""

    cls = pykeen.nn.modules.ProjEInteraction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
    )

    def _exp_score(self, h, r, t, d_e, d_r, b_c, b_p, activation) -> torch.FloatTensor:
        # f(h, r, t) = g(t z(D_e h + D_r r + b_c) + b_p)
        h, r, t = strip_dim(h, r, t)
        return (t * activation((d_e * h) + (d_r * r) + b_c)).sum() + b_p


class RESCALTests(InteractionTests, unittest.TestCase):
    """Tests for RESCAL interaction function."""

    cls = pykeen.nn.modules.RESCALInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        # f(h, r, t) = h @ r @ t
        h, r, t = strip_dim(h, r, t)
        return h.view(1, -1) @ r @ t.view(-1, 1)


class KG2ETests(InteractionTests, unittest.TestCase):
    """Tests for KG2E interaction function."""

    cls = pykeen.nn.modules.KG2EInteraction

    def _exp_score(self, exact, h_mean, h_var, r_mean, r_var, similarity, t_mean, t_var):
        assert similarity == "KL"
        h_mean, h_var, r_mean, r_var, t_mean, t_var = strip_dim(h_mean, h_var, r_mean, r_var, t_mean, t_var)
        e_mean, e_var = h_mean - t_mean, h_var + t_var
        p = torch.distributions.MultivariateNormal(loc=e_mean, covariance_matrix=torch.diag(e_var))
        q = torch.distributions.MultivariateNormal(loc=r_mean, covariance_matrix=torch.diag(r_var))
        return -torch.distributions.kl.kl_divergence(p, q)


class TuckerTests(InteractionTests, unittest.TestCase):
    """Tests for Tucker interaction function."""

    cls = pykeen.nn.modules.TuckerInteraction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
    )

    def _exp_score(self, bn_h, bn_hr, core_tensor, do_h, do_r, do_hr, h, r, t) -> torch.FloatTensor:
        # DO_{hr}(BN_{hr}(DO_h(BN_h(h)) x_1 DO_r(W x_2 r))) x_3 t
        h, r, t = strip_dim(h, r, t)
        a = do_r((core_tensor * r[None, :, None]).sum(dim=1, keepdims=True))  # shape: (embedding_dim, 1, embedding_dim)
        b = do_h(bn_h(h.view(1, -1))).view(-1)  # shape: (embedding_dim)
        c = (b[:, None, None] * a).sum(dim=0, keepdims=True)  # shape: (1, 1, embedding_dim)
        d = do_hr(bn_hr((c.view(1, -1)))).view(1, 1, -1)  # shape: (1, 1, 1, embedding_dim)
        return (d * t[None, None, :]).sum()


class RotatETests(InteractionTests, unittest.TestCase):
    """Tests for RotatE interaction function."""

    cls = pykeen.nn.modules.RotatEInteraction

    def _get_hrt(self, *shapes):  # noqa: D102
        # normalize length of r
        h, r, t = super()._get_hrt(*shapes)
        rc = view_complex(r)
        rl = (rc.abs() ** 2).sum(dim=-1).sqrt()
        r = r / rl.unsqueeze(dim=-1)
        return h, r, t

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        h, r, t = strip_dim(*(view_complex(x) for x in (h, r, t)))
        # check for unit length
        assert torch.allclose((r.abs() ** 2).sum(dim=-1).sqrt(), torch.ones(1))
        d = h * r - t
        return -(d.abs() ** 2).sum(dim=-1).sqrt()


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
        h, w_r, d_r, t = strip_dim(h, w_r, d_r, t)
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

    def _exp_score(self, h, r, m_r, t, p, power_norm) -> torch.FloatTensor:
        assert power_norm
        h, r, m_r, t = strip_dim(h, r, m_r, t)
        h_bot, t_bot = [clamp_norm(x.unsqueeze(dim=0) @ m_r, p=2, dim=-1, maxnorm=1.) for x in (h, t)]
        return -((h_bot + r - t_bot) ** p).sum()


class SETests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for SE interaction function."""

    cls = pykeen.nn.modules.StructuredEmbeddingInteraction

    def _exp_score(self, h, t, r_h, r_t, p, power_norm) -> torch.FloatTensor:
        assert not power_norm
        # -\|R_h h - R_t t\|
        h, t, r_h, r_t = strip_dim(h, t, r_h, r_t)
        h = r_h @ h.unsqueeze(dim=-1)
        t = r_t @ t.unsqueeze(dim=-1)
        return -(h - t).norm(p)


class UMTests(TranslationalInteractionTests, unittest.TestCase):
    """Tests for UM interaction function."""

    cls = pykeen.nn.modules.UnstructuredModelInteraction

    def _exp_score(self, h, t, p, power_norm) -> torch.FloatTensor:
        assert power_norm
        # -\|h - t\|
        h, t = strip_dim(h, t)
        return -(h - t).pow(p).sum()


class SimplEInteractionTests(InteractionTests, unittest.TestCase):
    """Tests for SimplE interaction function."""

    cls = pykeen.nn.modules.SimplEInteraction

    def _exp_score(self, h, r, t, h_inv, r_inv, t_inv, clamp) -> torch.FloatTensor:
        h, r, t, h_inv, r_inv, t_inv = strip_dim(h, r, t, h_inv, r_inv, t_inv)
        assert clamp is None
        return 0.5 * distmult_interaction(h, r, t) + 0.5 * distmult_interaction(h_inv, r_inv, t_inv)


class InteractionTestsTest(ptb.TestsTest[Interaction], unittest.TestCase):
    """Test for tests for all interaction functions."""

    base_cls = Interaction
    base_test = InteractionTests
    skip_cls = {
        TranslationalInteraction,
        LiteralInteraction,
    }
