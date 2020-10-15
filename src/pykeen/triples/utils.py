# -*- coding: utf-8 -*-

"""Instance creation utilities."""

from typing import Callable, Mapping, Optional, TextIO, Union

import numpy as np
from pkg_resources import iter_entry_points

from ..typing import LabeledTriples

__all__ = [
    'load_triples',
]


def _load_importers(group_subname: str) -> Mapping[str, Callable[[str], LabeledTriples]]:
    return {
        entry_point.name: entry_point.load()
        for entry_point in iter_entry_points(group=f'pykeen.triples.{group_subname}')
    }


#: Functions for specifying exotic resources with a given prefix
PREFIX_IMPORTERS: Mapping[str, Callable[[str], LabeledTriples]] = _load_importers('prefix_importer')
#: Functions for specifying exotic resources based on their file extension
EXTENSION_IMPORTERS: Mapping[str, Callable[[str], LabeledTriples]] = _load_importers('extension_importer')


def load_triples(path: Union[str, TextIO], delimiter: str = '\t', encoding: Optional[str] = None) -> LabeledTriples:
    """Load triples saved as tab separated values.

    Besides TSV handling, PyKEEN does not come with any importers pre-installed. A few can be found at:

    - :mod:`pybel.io.pykeen`
    - :mod:`bio2bel.io.pykeen`
    """
    if isinstance(path, str):
        for extension, handler in EXTENSION_IMPORTERS.items():
            if path.endswith(f'.{extension}'):
                return handler(path)

        for prefix, handler in PREFIX_IMPORTERS.items():
            if path.startswith(f'{prefix}:'):
                return handler(path[len(f'{prefix}:'):])

    if encoding is None:
        encoding = 'utf-8'

    return np.loadtxt(
        fname=path,
        dtype=str,
        comments='@Comment@ Head Relation Tail',
        delimiter=delimiter,
        encoding=encoding,
    )
