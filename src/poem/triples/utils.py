# -*- coding: utf-8 -*-

"""Instance creation utilities."""

import enum
from typing import TextIO, Union

import numpy as np

from ..typing import LabeledTriples

__all__ = [
    'load_triples',
    'Assumption',
]


def load_triples(path: Union[str, TextIO], delimiter='\t') -> LabeledTriples:
    """Load triples saved as tab separated values."""
    return np.loadtxt(
        fname=path,
        dtype=str,
        comments='@Comment@ Head Relation Tail',
        delimiter=delimiter,
    )


class Assumption(enum.Enum):
    """The assumption made by the model."""

    #: Local closed-world assumption (LCWA)
    local_closed = 'local-closed'

    #: Open world assumption (OWA)
    open = 'open'
