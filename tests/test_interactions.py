"""Tests for interaction functions."""
import unittest
from typing import Any, Generic, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union
from unittest.case import SkipTest

import torch

import pykeen.nn.modules
from pykeen.typing import Representation

T = TypeVar("T")


class GenericTests(Generic[T]):
    """Generic tests."""

    cls: Type[T]
    kwargs: Optional[Mapping[str, Any]] = None
    instance: T

    def setUp(self) -> None:
        kwargs = self.kwargs or {}
        kwargs = self._pre_instantiation_hook(kwargs=dict(kwargs))
        self.instance = self.cls(**kwargs)
        self.post_instantiation_hook()

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Perform actions before instantiation, potentially modyfing kwargs."""
        return kwargs

    def post_instantiation_hook(self) -> None:
        """Perform actions after instantiation."""


class InteractionTests(GenericTests[pykeen.nn.modules.InteractionFunction]):
    """Generic test for interaction functions."""

    dim: int = 2
    batch_size: int = 3
    num_relations: int = 5
    num_entities: int = 7

    def _get_hrt(
        self,
        *shapes: Tuple[int, ...],
        **kwargs
    ) -> Tuple[Union[Representation, Sequence[Representation]], ...]:
        kwargs.setdefault("d", self.dim)
        return tuple(
            torch.rand(*s, *(kwargs.get(ss) for ss in ms), requires_grad=True)
            for s, ms in zip(shapes, [self.cls.entity_shape, self.cls.relation_shape, self.cls.entity_shape])
        )

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

    def test_score_r(self):
        """Test score_r."""
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.num_relations,),
            (self.batch_size,),
        )
        scores = self.instance.score_r(h=h, all_relations=r, t=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, self.num_relations))

    def test_score_t(self):
        """Test score_t."""
        h, r, t = self._get_hrt(
            (self.batch_size,),
            (self.batch_size,),
            (self.num_entities,),
        )
        scores = self.instance.score_t(h=h, r=r, all_entities=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, self.num_entities))

    def test_forward(self):
        """Test forward."""
        for hs, rs, ts in [
            [
                (self.batch_size, 1),
                (1, self.num_relations),
                (self.batch_size, self.num_entities),
            ],
            [
                (1, 1),
                (1, self.num_relations),
                (self.batch_size, self.num_entities),
            ],
            [
                (1, self.num_entities),
                (1, self.num_relations),
                (1, self.num_entities),
            ],
        ]:
            with self.subTest(f"forward({hs}, {rs}, {ts})"):
                expected_shape = (max(hs[0], rs[0], ts[0]), hs[1], rs[1], ts[1])
                h, r, t = self._get_hrt(hs, rs, ts)
                scores = self.instance(h=h, r=r, t=t)
                self._check_scores(scores=scores, exp_shape=expected_shape)

    def test_scores(self):
        """Test individual scores."""
        self.instance.eval()
        for i in range(10):
            h, r, t = self._get_hrt((1, 1), (1, 1), (1, 1))
            scores = self.instance(h=h, r=r, t=t)
            exp_score = self._exp_score(h, r, t).item()
            assert scores.item() == exp_score

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        raise SkipTest("No score check implemented.")


class ComplExTests(InteractionTests, unittest.TestCase):
    """Tests for ComplEx interaction function."""

    cls = pykeen.nn.modules.ComplExInteractionFunction


class ConvETests(InteractionTests, unittest.TestCase):
    """Tests for ConvE interaction function."""

    cls = pykeen.nn.modules.ConvEInteractionFunction
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
        **kwargs
    ) -> Tuple[Union[Representation, Sequence[Representation]], ...]:
        h, r, t = super()._get_hrt(*shapes, **kwargs)
        t_bias = torch.rand_like(t[..., 0, None])
        return h, r, (t, t_bias)


class ConvKBTests(InteractionTests, unittest.TestCase):
    """Tests for ConvKB interaction function."""

    cls = pykeen.nn.modules.ConvKBInteractionFunction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
        num_filters=2 * InteractionTests.dim - 1,
    )


class DistMultTests(InteractionTests, unittest.TestCase):
    """Tests for DistMult interaction function."""

    cls = pykeen.nn.modules.DistMultInteractionFunction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        return (h * r * t).sum(dim=-1)


class ERMLPTests(InteractionTests, unittest.TestCase):
    """Tests for ERMLP interaction function."""

    cls = pykeen.nn.modules.ERMLPInteractionFunction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
        hidden_dim=2 * InteractionTests.dim - 1,
    )


class ERMLPETests(InteractionTests, unittest.TestCase):
    """Tests for ERMLP-E interaction function."""

    cls = pykeen.nn.modules.ERMLPInteractionFunction
    kwargs = dict(
        embedding_dim=InteractionTests.dim,
        hidden_dim=2 * InteractionTests.dim - 1,
    )


class HolETests(InteractionTests, unittest.TestCase):
    """Tests for HolE interaction function."""

    cls = pykeen.nn.modules.HolEInteractionFunction


class NTNTests(InteractionTests, unittest.TestCase):
    """Tests for NTN interaction function."""

    cls = pykeen.nn.modules.NTNInteractionFunction

    num_slices: int = 2

    def _get_hrt(
        self,
        *shapes: Tuple[int, ...],
    ) -> Tuple[Union[Representation, Sequence[Representation]], ...]:
        h, r, t = super()._get_hrt(*shapes)
        r_shape = r.shape[:-1]
        r = [
            torch.rand(*r_shape, *shape)
            for shape in [
                (self.num_slices, self.dim, self.dim),
                (self.num_slices,),
                (self.num_slices,),
                (self.num_slices, self.dim),
                (self.num_slices, self.dim),
            ]
        ]
        return h, r, t


class TransETests(InteractionTests, unittest.TestCase):
    """Tests for TransE interaction function."""

    cls = pykeen.nn.modules.TransEInteractionFunction
    kwargs = dict(
        p=2,
    )

    def _additional_score_checks(self, scores):
        assert (scores <= 0).all()

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        return -(h + r - t).norm(p=2, dim=-1)
