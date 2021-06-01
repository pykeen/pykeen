# -*- coding: utf-8 -*-

"""Instance creation utilities."""

import pathlib
from typing import Callable, Mapping, Optional, Sequence, Set, TextIO, Union

import numpy as np
import torch
from pkg_resources import iter_entry_points

from ..typing import LabeledTriples

__all__ = [
    'load_triples',
    'get_entities',
    'get_relations',
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


def load_triples(
    path: Union[str, pathlib.Path, TextIO],
    delimiter: str = '\t',
    encoding: Optional[str] = None,
    column_remapping: Optional[Sequence[int]] = None,
) -> LabeledTriples:
    """Load triples saved as tab separated values.

    :param path: The key for the data to be loaded. Typically, this will be a file path ending in ``.tsv``
        that points to a file with three columns - the head, relation, and tail. This can also be used to
        invoke PyKEEN data importer entrypoints (see below).
    :param delimiter: The delimiter between the columns in the file
    :param encoding: The encoding for the file. Defaults to utf-8.
    :param column_remapping: A remapping if the three columns do not follow the order head-relation-tail.
        For example, if the order is head-tail-relation, pass ``(0, 2, 1)``
    :returns: A numpy array representing "labeled" triples.

    :raises ValueError: if a column remapping was passed but it was not a length 3 sequence

    Besides TSV handling, PyKEEN does not come with any importers pre-installed. A few can be found at:

    - :mod:`pybel.io.pykeen`
    - :mod:`bio2bel.io.pykeen`
    """
    if isinstance(path, (str, pathlib.Path)):
        path = str(path)
        for extension, handler in EXTENSION_IMPORTERS.items():
            if path.endswith(f'.{extension}'):
                return handler(path)

        for prefix, handler in PREFIX_IMPORTERS.items():
            if path.startswith(f'{prefix}:'):
                return handler(path[len(f'{prefix}:'):])

    if encoding is None:
        encoding = 'utf-8'

    rv = np.loadtxt(
        fname=path,
        dtype=str,
        comments='@Comment@ Head Relation Tail',
        delimiter=delimiter,
        encoding=encoding,
    )
    if column_remapping is not None:
        if len(column_remapping) != 3:
            raise ValueError('remapping must have length of three')
        rv = rv[:, column_remapping]
    return rv


def get_entities(triples: torch.LongTensor) -> Set[int]:
    """Get all entities from the triples."""
    return set(triples[:, [0, 2]].flatten().tolist())


def get_relations(triples: torch.LongTensor) -> Set[int]:
    """Get all relations from the triples."""
    return set(triples[:, 1].tolist())
