# coding=utf-8
"""Generic unittests."""
import logging
import pprint
from typing import Any, Collection, Generic, Mapping, MutableMapping, Optional, Type, TypeVar

from pykeen.utils import get_all_subclasses, kwargs_or_empty

B = TypeVar('B')


class GenericTest(Generic[B]):
    """A generic test case."""

    #: The class
    cls: Type[B]

    #: The constructor keyword arguments
    kwargs: Optional[Mapping[str, Any]] = None

    #: The instance
    instance: B

    def setUp(self) -> None:
        """Instantiate the test instance.

        Do not override this method, but rather consider overriding the _pre_instantiation_hook or
         _post_instantiation_hook.
        """
        # Log everything
        logging.basicConfig(level=logging.DEBUG)
        self.kwargs = self._pre_instantiation_hook(kwargs=dict(kwargs_or_empty(kwargs=self.kwargs)))
        self.instance = self.cls(**self.kwargs)
        self._post_instantiation_hook()

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Prepare key-word based instantiation arguments."""
        return kwargs

    def _post_instantiation_hook(self) -> None:
        """Apply test preparations after the instance to test has been instantiated."""


class TestsTest(Generic[B]):
    """Checks whether all subclasses have unittests."""

    #: The base class to check for having unittests
    base_cls: Type[B]

    #: The base class for unittests
    base_test_cls: Type[GenericTest[B]]

    #: Blacklist some sub-classes of base_cls, e.g. since they are abstract
    skip_cls: Collection[Type[B]] = frozenset()

    def test_testing(self):
        """Check whether there are unittests for all sub-classes."""
        classes = get_all_subclasses(base_class=self.base_cls)
        tests = get_all_subclasses(base_class=self.base_test_cls)
        tested_classes = set(t.cls for t in tests if hasattr(t, 'cls'))
        uncovered_classes = classes.difference(tested_classes).difference(self.skip_cls)
        if len(uncovered_classes) > 0:
            raise NotImplementedError(f'No tests for \n{pprint.pformat(uncovered_classes)}')
