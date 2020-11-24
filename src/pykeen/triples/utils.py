# -*- coding: utf-8 -*-

"""Instance creation utilities."""

from typing import Callable, List, Mapping, Optional, TextIO, Union

import numpy as np
from pkg_resources import iter_entry_points
from tabulate import tabulate

from ..typing import LabeledTriples

__all__ = [
    'load_triples',
    'concatenate_triples_factories',
    'summarize',
    'calculate_ratios',
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


def concatenate_triples_factories(*triples_factories) -> np.ndarray:
    """Concatenate all of the triples over multiple triples factories."""
    return np.concatenate(
        [
            triples_factory.triples
            for triples_factory in triples_factories
        ],
        axis=0,
    )


def summary_str(training, testing, validation) -> str:
    """Make a summary string for the dataset."""
    headers = ['Set', 'Entities', 'Relations', 'Triples', 'Ratio']
    a, b, c = calculate_ratios(training, testing, validation)
    return tabulate(
        [
            ['Train', training.num_entities, training.num_relations, training.num_triples, a],
            ['Test', testing.num_entities, testing.num_relations, testing.num_triples, b],
            ['Valid', validation.num_entities, validation.num_relations, validation.num_triples, c],
        ],
        headers=headers,
    )


def summarize(training, testing, validation) -> None:
    """Summarize the dataset."""
    print(summary_str(training, testing, validation))


def calculate_ratios(*triples_factories) -> List[float]:
    """Calculate the ratios of the triples factories' sizes."""
    x = [tf.num_triples for tf in triples_factories]
    total_triples = sum(x)
    rv = [nt / total_triples for nt in x]
    rv[-1] = 1.0 - sum(rv[:-1])
    return rv
