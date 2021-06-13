# -*- coding: utf-8 -*-

"""Tests for interaction functions."""

import logging
from typing import Tuple
from unittest import SkipTest

import numpy
import torch
import unittest_templates

import pykeen.nn.modules
import pykeen.utils
from pykeen.nn.functional import _rotate_quaternion, _split_quaternion, distmult_interaction
from pykeen.nn.modules import FunctionalInteraction, Interaction, LiteralInteraction, TranslationalInteraction
from pykeen.utils import clamp_norm, project_entity, strip_dim, view_complex
from tests import cases

logger = logging.getLogger(__name__)


class ComplExTests(cases.InteractionTestCase):
    """Tests for ComplEx interaction function."""

    cls = pykeen.nn.modules.ComplExInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        h, r, t = [view_complex(x) for x in (h, r, t)]
        return (h * r * torch.conj(t)).sum().real


class ConvETests(cases.InteractionTestCase):
    """Tests for ConvE interaction function."""

    cls = pykeen.nn.modules.ConvEInteraction
    kwargs = dict(
        embedding_height=1,
        embedding_width=2,
        kernel_height=2,
        kernel_width=1,
        embedding_dim=cases.InteractionTestCase.dim,
    )

    def _get_hrt(
        self,
        *shapes: Tuple[int, ...],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:  # noqa: D102
        h, r, t = super()._get_hrt(*shapes)
        t_bias = torch.rand_like(t[..., 0, None])  # type: torch.FloatTensor
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


class ConvKBTests(cases.InteractionTestCase):
    """Tests for ConvKB interaction function."""

    cls = pykeen.nn.modules.ConvKBInteraction
    kwargs = dict(
        embedding_dim=cases.InteractionTestCase.dim,
        num_filters=2 * cases.InteractionTestCase.dim - 1,
    )

    def _exp_score(self, h, r, t, conv, activation, hidden_dropout, linear) -> torch.FloatTensor:  # noqa: D102
        # W_L drop(act(W_C \ast ([h; r; t]) + b_C)) + b_L
        # prepare conv input (N, C, H, W)
        x = torch.stack([x.view(-1) for x in (h, r, t)], dim=1).view(1, 1, -1, 3)
        x = conv(x)
        x = hidden_dropout(activation(x))
        return linear(x.view(1, -1))


class CrossETests(cases.InteractionTestCase):
    """Tests for CrossE interaction function."""

    cls = pykeen.nn.modules.CrossEInteraction
    kwargs = dict(
        embedding_dim=cases.InteractionTestCase.dim,
    )

    def _exp_score(self, h, r, c_r, t, bias, activation, dropout) -> torch.FloatTensor:  # noqa: D102
        h, r, c_r, t, bias = strip_dim(h, r, c_r, t, bias)
        return (dropout(activation(h * c_r + h * r * c_r + bias)) * t).sum()


class DistMultTests(cases.InteractionTestCase):
    """Tests for DistMult interaction function."""

    cls = pykeen.nn.modules.DistMultInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        return (h * r * t).sum(dim=-1)


class ERMLPTests(cases.InteractionTestCase):
    """Tests for ERMLP interaction function."""

    cls = pykeen.nn.modules.ERMLPInteraction
    kwargs = dict(
        embedding_dim=cases.InteractionTestCase.dim,
        hidden_dim=2 * cases.InteractionTestCase.dim - 1,
    )

    def _exp_score(self, h, r, t, hidden, activation, final) -> torch.FloatTensor:
        x = torch.cat([x.view(-1) for x in (h, r, t)])
        return final(activation(hidden(x)))


class ERMLPETests(cases.InteractionTestCase):
    """Tests for ERMLP-E interaction function."""

    cls = pykeen.nn.modules.ERMLPEInteraction
    kwargs = dict(
        embedding_dim=cases.InteractionTestCase.dim,
        hidden_dim=2 * cases.InteractionTestCase.dim - 1,
    )

    def _exp_score(self, h, r, t, mlp) -> torch.FloatTensor:  # noqa: D102
        x = torch.cat([x.view(1, -1) for x in (h, r)], dim=-1)
        return mlp(x).view(1, -1) @ t.view(-1, 1)


class HolETests(cases.InteractionTestCase):
    """Tests for HolE interaction function."""

    cls = pykeen.nn.modules.HolEInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        h, t = [torch.fft.rfft(x.view(1, -1), dim=-1) for x in (h, t)]
        h = torch.conj(h)
        c = torch.fft.irfft(h * t, n=h.shape[-1], dim=-1)
        return (c * r).sum()


class NTNTests(cases.InteractionTestCase):
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


class ProjETests(cases.InteractionTestCase):
    """Tests for ProjE interaction function."""

    cls = pykeen.nn.modules.ProjEInteraction
    kwargs = dict(
        embedding_dim=cases.InteractionTestCase.dim,
    )

    def _exp_score(self, h, r, t, d_e, d_r, b_c, b_p, activation) -> torch.FloatTensor:
        # f(h, r, t) = g(t z(D_e h + D_r r + b_c) + b_p)
        h, r, t = strip_dim(h, r, t)
        return (t * activation((d_e * h) + (d_r * r) + b_c)).sum() + b_p


class QuatETests(cases.InteractionTestCase):
    """Tests for QuatE interaction."""

    cls = pykeen.nn.modules.QuatEInteraction
    dim = 4 * cases.InteractionTestCase.dim  # quaternions

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        h, r, t = strip_dim(h, r, t)
        return -(_rotate_quaternion(*(_split_quaternion(x) for x in [h, r])) * t).sum()


class RESCALTests(cases.InteractionTestCase):
    """Tests for RESCAL interaction function."""

    cls = pykeen.nn.modules.RESCALInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        # f(h, r, t) = h @ r @ t
        h, r, t = strip_dim(h, r, t)
        return h.view(1, -1) @ r @ t.view(-1, 1)


class KG2ETests(cases.InteractionTestCase):
    """Tests for KG2E interaction function."""

    cls = pykeen.nn.modules.KG2EInteraction

    def _exp_score(self, exact, h_mean, h_var, r_mean, r_var, similarity, t_mean, t_var):
        assert similarity == "KL"
        h_mean, h_var, r_mean, r_var, t_mean, t_var = strip_dim(h_mean, h_var, r_mean, r_var, t_mean, t_var)
        e_mean, e_var = h_mean - t_mean, h_var + t_var
        p = torch.distributions.MultivariateNormal(loc=e_mean, covariance_matrix=torch.diag(e_var))
        q = torch.distributions.MultivariateNormal(loc=r_mean, covariance_matrix=torch.diag(r_var))
        return -torch.distributions.kl.kl_divergence(p, q)


class TuckerTests(cases.InteractionTestCase):
    """Tests for Tucker interaction function."""

    cls = pykeen.nn.modules.TuckerInteraction
    kwargs = dict(
        embedding_dim=cases.InteractionTestCase.dim,
    )

    def _exp_score(self, bn_h, bn_hr, core_tensor, do_h, do_r, do_hr, h, r, t) -> torch.FloatTensor:
        # DO_{hr}(BN_{hr}(DO_h(BN_h(h)) x_1 DO_r(W x_2 r))) x_3 t
        h, r, t = strip_dim(h, r, t)
        a = do_r((core_tensor * r[None, :, None]).sum(dim=1, keepdims=True))  # shape: (embedding_dim, 1, embedding_dim)
        b = do_h(bn_h(h.view(1, -1))).view(-1)  # shape: (embedding_dim)
        c = (b[:, None, None] * a).sum(dim=0, keepdims=True)  # shape: (1, 1, embedding_dim)
        d = do_hr(bn_hr((c.view(1, -1)))).view(1, 1, -1)  # shape: (1, 1, 1, embedding_dim)
        return (d * t[None, None, :]).sum()


class RotatETests(cases.InteractionTestCase):
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


class TransDTests(cases.TranslationalInteractionTests):
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


class TransETests(cases.TranslationalInteractionTests):
    """Tests for TransE interaction function."""

    cls = pykeen.nn.modules.TransEInteraction

    def _exp_score(self, h, r, t, p, power_norm) -> torch.FloatTensor:
        assert not power_norm
        return -(h + r - t).norm(p=p, dim=-1)


class TransHTests(cases.TranslationalInteractionTests):
    """Tests for TransH interaction function."""

    cls = pykeen.nn.modules.TransHInteraction

    def _exp_score(self, h, w_r, d_r, t, p, power_norm) -> torch.FloatTensor:  # noqa: D102
        assert not power_norm
        h, w_r, d_r, t = strip_dim(h, w_r, d_r, t)
        h, t = [x - (x * w_r).sum() * w_r for x in (h, t)]
        return -(h + d_r - t).norm(p=p)


class TransRTests(cases.TranslationalInteractionTests):
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


class SETests(cases.TranslationalInteractionTests):
    """Tests for SE interaction function."""

    cls = pykeen.nn.modules.StructuredEmbeddingInteraction

    def _exp_score(self, h, t, r_h, r_t, p, power_norm) -> torch.FloatTensor:
        assert not power_norm
        # -\|R_h h - R_t t\|
        h, t, r_h, r_t = strip_dim(h, t, r_h, r_t)
        h = r_h @ h.unsqueeze(dim=-1)
        t = r_t @ t.unsqueeze(dim=-1)
        return -(h - t).norm(p)


class UMTests(cases.TranslationalInteractionTests):
    """Tests for UM interaction function."""

    cls = pykeen.nn.modules.UnstructuredModelInteraction

    def _exp_score(self, h, t, p, power_norm) -> torch.FloatTensor:
        assert power_norm
        # -\|h - t\|
        h, t = strip_dim(h, t)
        return -(h - t).pow(p).sum()


class PairRETests(cases.TranslationalInteractionTests):
    """Tests for PairRE interaction function."""

    cls = pykeen.nn.modules.PairREInteraction

    def _exp_score(self, h, r_h, r_t, t, p: float, power_norm: bool) -> torch.FloatTensor:
        s = (h * r_h - t * r_t).norm(p)
        if power_norm:
            s = s.pow(p)
        return -s


class SimplEInteractionTests(cases.InteractionTestCase):
    """Tests for SimplE interaction function."""

    cls = pykeen.nn.modules.SimplEInteraction

    def _exp_score(self, h, r, t, h_inv, r_inv, t_inv, clamp) -> torch.FloatTensor:
        h, r, t, h_inv, r_inv, t_inv = strip_dim(h, r, t, h_inv, r_inv, t_inv)
        assert clamp is None
        return 0.5 * distmult_interaction(h, r, t) + 0.5 * distmult_interaction(h_inv, r_inv, t_inv)


class MuRETests(cases.TranslationalInteractionTests):
    """Tests for MuRE interaction function."""

    cls = pykeen.nn.modules.MuREInteraction

    def _exp_score(self, h, b_h, r_vec, r_mat, t, b_t, p, power_norm) -> torch.FloatTensor:
        s = (h * r_mat) + r_vec - t
        s = s.norm(p=p)
        if power_norm:
            s = s.pow(p)
        s = -s
        s = s + b_h + b_t
        return s

    def _additional_score_checks(self, scores):
        # Since MuRE has offsets, the scores do not need to negative
        pass


class MonotonicAffineTransformationInteractionTests(cases.InteractionTestCase):
    """Tests for monotonic affine transformation interaction adapter."""

    cls = pykeen.nn.modules.MonotonicAffineTransformationInteraction
    kwargs = dict(
        base=pykeen.nn.modules.TransEInteraction(p=2),
    )

    def test_scores(self):  # noqa: D102
        raise SkipTest("Not a functional interaction.")

    def _exp_score(self, **kwargs) -> torch.FloatTensor:  # noqa: D102
        # We do not need this, since we do not check for functional consistency anyway
        raise NotImplementedError

    def test_monotonicity(self):
        """Verify monotonicity."""
        for hs, rs, ts in self._get_test_shapes():
            h, r, t = self._get_hrt(hs, rs, ts)
            s_t = self.instance(h=h, r=r, t=t).view(-1)
            s_o = self.instance.base(h=h, r=r, t=t).view(-1)
            # intra-interaction comparison
            c_t = (s_t.unsqueeze(dim=0) > s_t.unsqueeze(dim=1))
            c_o = (s_o.unsqueeze(dim=0) > s_o.unsqueeze(dim=1))
            assert (c_t == c_o).all()


class InteractionTestsTestCase(unittest_templates.MetaTestCase[Interaction]):
    """Test for tests for all interaction functions."""

    base_cls = Interaction
    base_test = cases.InteractionTestCase
    skip_cls = {
        Interaction,
        FunctionalInteraction,
        TranslationalInteraction,
        LiteralInteraction,
    }
