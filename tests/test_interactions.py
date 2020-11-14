"""Tests for interaction functions."""
import unittest
from typing import Any, Generic, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar
from unittest.case import SkipTest

import torch

from pykeen.nn.modules import DistMultInteractionFunction, InteractionFunction, TransEInteractionFunction
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


class InteractionTests(GenericTests[InteractionFunction]):
    """Generic test for interaction functions."""

    dim: int = 2
    batch_size: int = 3
    num_relations: int = 5
    num_entities: int = 7

    def _get_hrt(
        self,
        *shapes: Tuple[int, ...],
    ) -> Tuple[Representation, ...]:
        return tuple([torch.rand(*s, requires_grad=True) for s in shapes])

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
            (self.batch_size, self.dim),
            (self.batch_size, self.dim),
            (self.batch_size, self.dim),
        )
        scores = self.instance.score_hrt(h=h, r=r, t=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, 1))

    def test_score_h(self):
        """Test score_h."""
        h, r, t = self._get_hrt(
            (self.num_entities, self.dim),
            (self.batch_size, self.dim),
            (self.batch_size, self.dim),
        )
        scores = self.instance.score_h(all_entities=h, r=r, t=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, self.num_entities))

    def test_score_r(self):
        """Test score_r."""
        h, r, t = self._get_hrt(
            (self.batch_size, self.dim),
            (self.num_relations, self.dim),
            (self.batch_size, self.dim),
        )
        scores = self.instance.score_r(h=h, all_relations=r, t=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, self.num_relations))

    def test_score_t(self):
        """Test score_t."""
        h, r, t = self._get_hrt(
            (self.batch_size, self.dim),
            (self.batch_size, self.dim),
            (self.num_entities, self.dim),
        )
        scores = self.instance.score_t(h=h, r=r, all_entities=t)
        self._check_scores(scores=scores, exp_shape=(self.batch_size, self.num_entities))

    def test_forward(self):
        """Test forward."""
        for hs, rs, ts in [
            [
                (self.batch_size, 1, self.dim),
                (1, self.num_relations, self.dim),
                (self.batch_size, self.num_entities, self.dim),
            ],
            [
                (1, 1, self.dim),
                (1, self.num_relations, self.dim),
                (self.batch_size, self.num_entities, self.dim),
            ],
            [
                (1, self.num_entities, self.dim),
                (1, self.num_relations, self.dim),
                (1, self.num_entities, self.dim),
            ],
        ]:
            with self.subTest(f"forward({hs}, {rs}, {ts})"):
                expected_shape = (max(hs[0], rs[0], ts[0]), hs[1], rs[1], ts[1])
                h, r, t = self._get_hrt(hs, rs, ts)
                scores = self.instance(h=h, r=r, t=t)
                self._check_scores(scores=scores, exp_shape=expected_shape)

    def test_scores(self):
        """Test individual scores."""
        for i in range(10):
            h, r, t = self._get_hrt((1, 1, self.dim), (1, 1, self.dim), (1, 1, self.dim))
            scores = self.instance(h=h, r=r, t=t)
            exp_score = self._exp_score(h, r, t).item()
            assert scores.item() == exp_score

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        raise SkipTest()


class DistMultTests(InteractionTests, unittest.TestCase):
    """Tests for DistMult interaction function."""

    cls = DistMultInteractionFunction

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        return (h * r * t).sum(dim=-1)


class TransETests(InteractionTests, unittest.TestCase):
    """Tests for TransE interaction function."""

    cls = TransEInteractionFunction
    kwargs = dict(
        p=2,
    )

    def _additional_score_checks(self, scores):
        assert (scores <= 0).all()

    def _exp_score(self, h, r, t) -> torch.FloatTensor:
        return -(h + r - t).norm(p=2, dim=-1)
