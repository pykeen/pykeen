# -*- coding: utf-8 -*-

"""Assumptions supported by POEM."""

import enum

__all__ = [
    'Assumption',
]


class Assumption(enum.Enum):
    """The assumption made by the model."""

    #: Closed-world assumption (CWA)
    closed = 'closed'

    #: Open world assumption (OWA)
    open = 'open'
