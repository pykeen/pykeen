# -*- coding: utf-8 -*-

"""Instance creation utilities."""

from typing import Callable, Mapping, Optional, Set, TextIO, Union

import numpy as np
from pkg_resources import iter_entry_points

from ..typing import LabeledTriples, RandomHint
from ..utils import ensure_random_state

__all__ = [
    'load_triples',
    'generate_triples',
    'get_entities',
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


def generate_triples(
    num_entities: int = 33,
    num_relations: int = 7,
    num_triples: int = 101,
    compact: bool = True,
    random_state: RandomHint = None,
) -> np.ndarray:
    """Generate random triples."""
    random_state = ensure_random_state(random_state)
    rv = np.stack([
        random_state.randint(num_entities, size=(num_triples,)),
        random_state.randint(num_relations, size=(num_triples,)),
        random_state.randint(num_entities, size=(num_triples,)),
    ], axis=1)

    if compact:
        new_entity_id = {
            entity: i
            for i, entity in enumerate(sorted(get_entities(rv)))
        }
        new_relation_id = {
            relation: i
            for i, relation in enumerate(sorted(get_relations(rv)))
        }
        rv = np.asarray([
            [new_entity_id[h], new_relation_id[r], new_entity_id[t]]
            for h, r, t in rv
        ])

    return rv


def get_entities(triples) -> Set:
    """Get all entities from the triples."""
    return set(triples[:, [0, 2]].flatten().tolist())


def get_relations(triples) -> Set:
    """Get all relations from the triples."""
    return set(triples[:, 1])
