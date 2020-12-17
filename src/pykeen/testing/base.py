# -*- coding: utf-8 -*-

"""Base classes for simplified testing."""

import unittest
from typing import Any, Collection, Generic, Mapping, MutableMapping, Optional, Type, TypeVar

from ..utils import get_subclasses, set_random_seed

__all__ = [
    'GenericTestCase',
    'TestsTestCase',
]

T = TypeVar("T")


class GenericTestCase(Generic[T]):
    """Generic tests."""

    cls: Type[T]
    kwargs: Optional[Mapping[str, Any]] = None
    instance: T

    def setUp(self) -> None:
        """Set up the generic testing method."""
        # fix seeds for reproducibility
        set_random_seed(seed=42)
        kwargs = self.kwargs or {}
        kwargs = self._pre_instantiation_hook(kwargs=dict(kwargs))
        self.instance = self.cls(**kwargs)
        self.post_instantiation_hook()

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Perform actions before instantiation, potentially modyfing kwargs."""
        return kwargs

    def post_instantiation_hook(self) -> None:
        """Perform actions after instantiation."""


class TestsTestCase(Generic[T], unittest.TestCase):
    """A generic test for tests."""

    base_cls: Type[T]
    base_test: Type[GenericTestCase[T]]
    skip_cls: Collection[T] = tuple()

    def test_testing(self):
        """Check that there is a test for all subclasses."""
        to_test = set(get_subclasses(self.base_cls)).difference(self.skip_cls)
        tested = (test_cls.cls for test_cls in get_subclasses(self.base_test) if hasattr(test_cls, "cls"))
        not_tested = to_test.difference(tested)
        assert not not_tested, not_tested
