"""Tests for interaction functions."""
import unittest
from typing import Any, Generic, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar

import torch

from pykeen.nn.modules import InteractionFunction, TransEInteractionFunction
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

    def _get_shapes_for_score_(self, dim: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int, int]]:
        result = []
        num_choices = self.num_entities if dim != 1 else self.num_relations
        for i in range(3):
            shape = [1, 1, 1, 1]
            if i == dim:
                shape[i + 1] = num_choices
            else:
                shape[0] = self.batch_size
            result.append(tuple(shape))
        shape = [self.batch_size, 1, 1, 1]
        shape[dim + 1] = num_choices
        result.append(tuple(shape))
        return tuple(result)

    def test_forward(self):
        for hs, rs, ts, exp in [
            # slcwa
            [
                (self.batch_size, 1, self.dim),
                (self.batch_size, 1, self.dim),
                (self.batch_size, 1, self.dim),
                (self.batch_size, 1, 1, 1),
            ],
            # score_h
            self._get_shapes_for_score_(dim=0),
            # score_r
            self._get_shapes_for_score_(dim=1),
            # score_t
            self._get_shapes_for_score_(dim=2),
        ]:
            h, r, t = self._get_hrt(hs, rs, ts)
            scores = self.instance.forward(h=h, r=r, t=t)
            assert torch.is_tensor(scores)
            assert scores.dtype == torch.float32
            assert scores.ndimension() == 4
            assert scores.shape == exp
            assert scores.requires_grad


class TransETests(InteractionTests, unittest.TestCase):
    """Tests for TransE interaction function."""

    cls = TransEInteractionFunction
    kwargs = dict(
        p=2,
    )
