# -*- coding: utf-8 -*-

"""Tests for interaction functions."""

import logging
import unittest
from typing import Any, MutableMapping, Sequence, Tuple, Union
from unittest import SkipTest

import numpy
import torch
import torch.nn.functional
import unittest_templates
from torch import nn

import pykeen.nn.modules
import pykeen.utils
from pykeen.models.unimodal.quate import quaternion_normalizer
from pykeen.nn.functional import distmult_interaction
from pykeen.typing import Representation, Sign
from pykeen.utils import clamp_norm, complex_normalize, einsum, ensure_tuple, project_entity
from tests import cases

logger = logging.getLogger(__name__)


class ComplExTests(cases.InteractionTestCase):
    """Tests for ComplEx interaction function."""

    cls = pykeen.nn.modules.ComplExInteraction
    dtype = torch.cfloat

    # TODO: we could move this part into the interaction module itself
    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
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

    def _exp_score(
        self,
        embedding_height,
        embedding_width,
        h,
        hr1d,
        hr2d,
        input_channels,
        r,
        t,
        t_bias,
    ) -> torch.FloatTensor:
        x = torch.cat(
            [
                h.view(1, input_channels, embedding_height, embedding_width),
                r.view(1, input_channels, embedding_height, embedding_width),
            ],
            dim=2,
        )
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


class CPInteractionTests(cases.InteractionTestCase):
    """Test for the canonical tensor decomposition interaction."""

    cls = pykeen.nn.modules.CPInteraction
    shape_kwargs = dict(
        k=3,
    )

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        return (h * r * t).sum(dim=(-2, -1))


class CrossETests(cases.InteractionTestCase):
    """Tests for CrossE interaction function."""

    cls = pykeen.nn.modules.CrossEInteraction
    kwargs = dict(
        embedding_dim=cases.InteractionTestCase.dim,
    )

    def _exp_score(self, h, r, c_r, t, bias, activation, dropout) -> torch.FloatTensor:  # noqa: D102
        return (dropout(activation(h * c_r + h * r * c_r + bias)) * t).sum()


class DistMultTests(cases.InteractionTestCase):
    """Tests for DistMult interaction function."""

    cls = pykeen.nn.modules.DistMultInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        return (h * r * t).sum(dim=-1)


class DistMATests(cases.InteractionTestCase):
    """Tests for DistMA interaction function."""

    cls = pykeen.nn.modules.DistMAInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        return (h * r).sum() + (r * t).sum() + (h * t).sum()


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
        score = 0.0
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
        return (t * activation((d_e * h) + (d_r * r) + b_c)).sum() + b_p


def _rotate_quaternion(qa: torch.FloatTensor, qb: torch.FloatTensor) -> torch.FloatTensor:
    # Rotate (=Hamilton product in quaternion space).
    return torch.stack(
        [
            qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3],
            qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2],
            qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1],
            qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0],
        ],
        dim=-1,
    )


class QuatETests(cases.InteractionTestCase):
    """Tests for QuatE interaction."""

    cls = pykeen.nn.modules.QuatEInteraction
    shape_kwargs = dict(k=4)  # quaternions
    atol = 1.0e-06

    def _exp_score(
        self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, table: torch.Tensor
    ) -> torch.FloatTensor:  # noqa: D102
        # we calculate the scores using the hard-coded formula, instead of utilizing table + einsum
        x = _rotate_quaternion(*(x.unbind(dim=-1) for x in [h, r]))
        return -(x * t).sum()

    def _get_hrt(self, *shapes):
        h, r, t = super()._get_hrt(*shapes)
        r = quaternion_normalizer(r)
        return h, r, t


class RESCALTests(cases.InteractionTestCase):
    """Tests for RESCAL interaction function."""

    cls = pykeen.nn.modules.RESCALInteraction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        # f(h, r, t) = h @ r @ t
        return h.view(1, -1) @ r @ t.view(-1, 1)


class KG2ETests(cases.InteractionTestCase):
    """Tests for KG2E interaction function."""

    cls = pykeen.nn.modules.KG2EInteraction

    def _exp_score(self, exact, h_mean, h_var, r_mean, r_var, similarity, t_mean, t_var):
        assert similarity == "KL"
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
        a = do_r((core_tensor * r[None, :, None]).sum(dim=1, keepdims=True))  # shape: (embedding_dim, 1, embedding_dim)
        b = do_h(bn_h(h.view(1, -1))).view(-1)  # shape: (embedding_dim)
        c = (b[:, None, None] * a).sum(dim=0, keepdims=True)  # shape: (1, 1, embedding_dim)
        d = do_hr(bn_hr((c.view(1, -1)))).view(1, 1, -1)  # shape: (1, 1, 1, embedding_dim)
        return (d * t[None, None, :]).sum()


class RotatETests(cases.InteractionTestCase):
    """Tests for RotatE interaction function."""

    cls = pykeen.nn.modules.RotatEInteraction
    dtype = torch.cfloat

    def _get_hrt(self, *shapes):  # noqa: D102
        h, r, t = super()._get_hrt(*shapes)
        # normalize rotations to unit modulus
        r = complex_normalize(r)
        return h, r, t

    def _exp_score(self, h, r, t) -> torch.FloatTensor:  # noqa: D102
        # check for unit modulus
        assert torch.allclose(r.abs(), torch.ones_like(r.abs()))
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
        h = t = torch.as_tensor(data=[2.0, 2.0], dtype=torch.float).view(1, 2)
        h_p = t_p = torch.as_tensor(data=[3.0, 3.0], dtype=torch.float).view(1, 2)

        # relation embeddings
        r = torch.as_tensor(data=[4.0], dtype=torch.float).view(1, 1)
        r_p = torch.as_tensor(data=[5.0], dtype=torch.float).view(1, 1)

        # Compute Scores
        scores = self.instance.score_hrt(h=(h, h_p), r=(r, r_p), t=(t, t_p))
        first_score = scores[0].item()
        self.assertAlmostEqual(first_score, -16, delta=0.01)

    def test_manual_big_relation_dim(self):
        """Manually test the value of the interaction function."""
        # entity embeddings
        h = t = torch.as_tensor(data=[2.0, 2.0], dtype=torch.float).view(1, 2)
        h_p = t_p = torch.as_tensor(data=[3.0, 3.0], dtype=torch.float).view(1, 2)

        # relation embeddings
        r = torch.as_tensor(data=[3.0, 3.0, 3.0], dtype=torch.float).view(1, 3)
        r_p = torch.as_tensor(data=[4.0, 4.0, 4.0], dtype=torch.float).view(1, 3)

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
        h_bot, t_bot = [clamp_norm(x.unsqueeze(dim=0) @ m_r, p=2, dim=-1, maxnorm=1.0) for x in (h, t)]
        return -((h_bot + r - t_bot) ** p).sum()


class SETests(cases.TranslationalInteractionTests):
    """Tests for SE interaction function."""

    cls = pykeen.nn.modules.SEInteraction

    def _exp_score(self, h, t, r_h, r_t, p, power_norm) -> torch.FloatTensor:
        assert not power_norm
        # -\|R_h h - R_t t\|
        h = r_h @ h.unsqueeze(dim=-1)
        t = r_t @ t.unsqueeze(dim=-1)
        return -(h - t).norm(p)


class UMTests(cases.TranslationalInteractionTests):
    """Tests for UM interaction function."""

    cls = pykeen.nn.modules.UMInteraction

    def _exp_score(self, h, t, p, power_norm) -> torch.FloatTensor:
        assert power_norm
        # -\|h - t\|
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


class TorusETests(cases.TranslationalInteractionTests):
    """Tests for the TorusE interaction function."""

    cls = pykeen.nn.modules.TorusEInteraction

    def _exp_score(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        p: Union[int, str] = 2,
        power_norm: bool = False,
    ) -> torch.FloatTensor:
        assert not power_norm
        d = h + r - t
        d = d - torch.floor(d)
        d = torch.minimum(d, 1.0 - d)
        return -d.norm(p=p)


class TransFTests(cases.InteractionTestCase):
    """Tests for the TransF interaction function."""

    cls = pykeen.nn.modules.TransFInteraction

    def _exp_score(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        left = ((h + r) * t).sum(dim=-1)
        right = (h * (t - r)).sum(dim=-1)
        return left + right


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
            c_t = s_t.unsqueeze(dim=0) > s_t.unsqueeze(dim=1)
            c_o = s_o.unsqueeze(dim=0) > s_o.unsqueeze(dim=1)
            assert (c_t == c_o).all()


class TransformerTests(cases.InteractionTestCase):
    """Tests for the Transformer interaction function."""

    cls = pykeen.nn.modules.TransformerInteraction
    # dimension needs to be divisible by num_heads
    dim = 8
    kwargs = dict(
        num_heads=2,
        dim_feedforward=7,
    )

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["input_dim"] = self.dim
        assert self.dim % kwargs["num_heads"] == 0
        return kwargs

    def _exp_score(
        self,
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        transformer: nn.TransformerEncoder,
        position_embeddings: torch.FloatTensor,
        final: nn.Module,
    ) -> torch.FloatTensor:  # noqa: D102
        x = torch.stack([h, r], dim=0) + position_embeddings
        x = transformer(src=x.unsqueeze(dim=1))
        x = x.sum(dim=0)
        x = final(x).squeeze(dim=0)
        return (x * t).sum()


class MultiLinearTuckerInteractionTests(cases.InteractionTestCase):
    """Tests for multi-linear TuckER."""

    cls = pykeen.nn.modules.MultiLinearTuckerInteraction
    shape_kwargs = dict(e=3, f=5)

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs)
        kwargs["head_dim"] = self.dim
        kwargs["relation_dim"] = self.shape_kwargs["e"]
        kwargs["tail_dim"] = self.shape_kwargs["f"]
        return kwargs

    def _exp_score(self, core_tensor, h, r, t) -> torch.FloatTensor:
        return einsum("ijk,i,j,k", core_tensor, h, r, t)


class InteractionTestsTestCase(unittest_templates.MetaTestCase[pykeen.nn.modules.Interaction]):
    """Test for tests for all interaction functions."""

    base_cls = pykeen.nn.modules.Interaction
    base_test = cases.InteractionTestCase
    skip_cls = {
        pykeen.nn.modules.Interaction,
        pykeen.nn.modules.FunctionalInteraction,
        pykeen.nn.modules.NormBasedInteraction,
        # FIXME
        pykeen.nn.modules.BoxEInteraction,
    }


class ParallelSliceBatchesTest(unittest.TestCase):
    """Tests for parallel_slice_batches."""

    def _verify(
        self,
        z: Representation,
        z_batch: Representation,
        dim: int,
        split_size: int,
    ) -> None:
        """Verify a sliced representations."""
        if torch.is_tensor(z):
            assert torch.is_tensor(z_batch)
            assert z.ndim == z_batch.ndim
            for i, (d, d_batch) in enumerate(zip(z.shape, z_batch.shape)):
                if i == dim:
                    assert d_batch <= split_size
                    assert d_batch <= d
                else:
                    assert d_batch == d
        else:
            assert not torch.is_tensor(z_batch)
            assert len(z) == len(z_batch)
            for y, y_batch in zip(z, z_batch):
                self._verify(z=y, z_batch=y_batch, dim=dim, split_size=split_size)

    def _generate(
        self,
        shape: Union[Tuple[int, ...], Sequence[Tuple[int, ...]]],
    ) -> Representation:
        """Generate dummy representations for the given shape(s)."""
        if not shape:
            return []
        if isinstance(shape[0], tuple):
            # multiple
            return [self._generate(s) for s in shape]
        # single
        return torch.empty(size=shape, device="meta")

    def _test(
        self,
        h_shape: Union[Tuple[int, ...], Sequence[Tuple[int, ...]]],
        r_shape: Union[Tuple[int, ...], Sequence[Tuple[int, ...]]],
        t_shape: Union[Tuple[int, ...], Sequence[Tuple[int, ...]]],
        dim: int,
        split_size: int,
    ):
        """Test slicing for given representation shapes."""
        h, r, t = [self._generate(s) for s in (h_shape, r_shape, t_shape)]
        for batch in pykeen.nn.modules.parallel_slice_batches(h, r, t, split_size=split_size, dim=dim):
            assert len(batch) == 3
            for old, new in zip((h, r, t), batch):
                self._verify(old, new, dim, split_size)

    def test_score_t(self):
        """Test parallel_slice_batches with single representations."""
        s, b, n, d = 2, 3, 7, 5
        self._test(h_shape=(b, 1, d), r_shape=(b, 1, d), t_shape=(1, n, d), dim=1, split_size=s)

    def test_multiple(self):
        """Test parallel_slice_batches with a multiple representations."""
        s, b, n, d = 2, 3, 7, 5
        self._test(h_shape=((b, 1, d), (b, 1, d, d)), r_shape=(b, 1, d), t_shape=(1, n, d), dim=1, split_size=s)
        self._test(h_shape=(b, 1, d), r_shape=(b, 1, d), t_shape=((1, n, d), (1, n, 2 * d)), dim=1, split_size=s)

    def test_empty(self):
        """Test parallel_slice_batches with missing relation representations."""
        s, b, n, d = 2, 3, 7, 5
        self._test(h_shape=(b, 1, d), r_shape=[], t_shape=(1, n, d), dim=1, split_size=s)


class TripleRETests(cases.TranslationalInteractionTests):
    """Tests for TripleRE interaction function."""

    cls = pykeen.nn.modules.TripleREInteraction

    def _exp_score(self, h, r_head, r_mid, r_tail, t, u, p, power_norm) -> torch.FloatTensor:  # noqa: D102
        assert not power_norm
        if u is None:
            u = 0.0
        #  head * (re_head + self.u * e_h) - tail * (re_tail + self.u * e_t) + re_mid
        return -(h * (r_head + u * torch.ones_like(r_head)) - t * (r_tail + u * torch.ones_like(r_tail)) + r_mid).norm(
            p=p,
        )


class AutoSFTests(cases.InteractionTestCase):
    """Tests for the AutoSF interaction function."""

    cls = pykeen.nn.modules.AutoSFInteraction
    kwargs = dict(
        coefficients=(
            (0, 0, 0, 1),
            (1, 1, 1, -1),
        ),
    )

    def _exp_score(
        self,
        h: Sequence[torch.FloatTensor],
        r: Sequence[torch.FloatTensor],
        t: Sequence[torch.FloatTensor],
        coefficients: Sequence[Tuple[int, int, int, Sign]],
    ) -> torch.FloatTensor:  # noqa: D102
        h, r, t = ensure_tuple(h, r, t)
        return sum(s * (h[i] * r[j] * t[k]).sum(dim=-1) for i, j, k, s in coefficients)


class LineaRETests(cases.TranslationalInteractionTests):
    """Test for LineaRE interaction."""

    cls = pykeen.nn.modules.LineaREInteraction

    def _exp_score(self, h, r_head, r_mid, r_tail, t, p, power_norm) -> torch.FloatTensor:
        s = h * r_head - t * r_tail + r_mid
        if power_norm:
            s = s.pow(p).sum(dim=-1)
        else:
            s = s.norm(p=p)
        return -s
