# -*- coding: utf-8 -*-

"""Instance creation utilities."""

import pathlib
from typing import Callable, Mapping, Optional, Sequence, Set, TextIO, Union

import numpy as np
import pandas
import torch
from pkg_resources import iter_entry_points

from ..typing import LabeledTriples

__all__ = [
    'load_triples',
    'get_entities',
    'get_relations',
    'tensor_to_df',
]

TRIPLES_DF_COLUMNS = ('head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label')


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


def tensor_to_df(
    tensor: torch.LongTensor,
    **kwargs: Union[torch.Tensor, np.ndarray, Sequence],
) -> pandas.DataFrame:
    """Take a tensor of triples and make a pandas dataframe with labels.

    :param tensor: shape: (n, 3)
        The triples, ID-based and in format (head_id, relation_id, tail_id).
    :param kwargs:
        Any additional number of columns. Each column needs to be of shape (n,). Reserved column names:
        {"head_id", "head_label", "relation_id", "relation_label", "tail_id", "tail_label"}.

    :return:
        A dataframe with n rows, and 3 + len(kwargs) columns.

    :raises ValueError:
        If a reserved column name appears in kwargs.
    """
    # Input validation
    additional_columns = set(kwargs.keys())
    forbidden = additional_columns.intersection(TRIPLES_DF_COLUMNS)
    if len(forbidden) > 0:
        raise ValueError(
            f'The key-words for additional arguments must not be in {TRIPLES_DF_COLUMNS}, but {forbidden} were '
            f'used.',
        )

    # convert to numpy
    tensor = tensor.cpu().numpy()
    data = dict(zip(['head_id', 'relation_id', 'tail_id'], tensor.T))

    # Additional columns
    for key, values in kwargs.items():
        # convert PyTorch tensors to numpy
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        data[key] = values

    # convert to dataframe
    rv = pandas.DataFrame(data=data)

    # Re-order columns
    columns = list(TRIPLES_DF_COLUMNS[::2]) + sorted(set(rv.columns).difference(TRIPLES_DF_COLUMNS))
    return rv.loc[:, columns]
