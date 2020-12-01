# -*- coding: utf-8 -*-

"""An adapter for Neptune.ai."""

from typing import TYPE_CHECKING

from .base import ResultTracker

if TYPE_CHECKING:
    import neptune  # noqa

__all__ = [
    'NeptuneResultTracker',
]


class NeptuneResultTracker(ResultTracker):
    """A tracker for Neptune.ai."""

    def __init__(self):
        import neptune as _neptune
        self.neptune = _neptune
