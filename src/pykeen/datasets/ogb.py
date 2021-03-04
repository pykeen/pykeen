# -*- coding: utf-8 -*-

"""Load the OGB datasets.

Run with ``python -m pykeen.datasets.ogb``
"""

from typing import ClassVar, Optional

import click
import numpy as np
from docdata import parse_docdata
from more_click import verbose_option

from .base import LazyDataset
from ..triples import TriplesFactory

__all__ = [
    'OGBLoader',
    'OGBBioKG',
    'OGBWikiKG',
]


class OGBLoader(LazyDataset):
    """Load from the Open Graph Benchmark (OGB)."""

    #: The name of the dataset to download
    name: ClassVar[str]

    def __init__(self, cache_root: Optional[str] = None, create_inverse_triples: bool = False):
        """Initialize the OGB loader.

        :param cache_root: An optional override for where data should be cached.
            If not specified, uses default PyKEEN location with :mod:`pystow`.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        """
        self.cache_root = self._help_cache(cache_root)
        self.create_inverse_triples = create_inverse_triples

    def _load(self) -> None:
        try:
            from ogb.linkproppred import LinkPropPredDataset
        except ImportError as e:
            raise ModuleNotFoundError(
                f'Need to `pip install ogb` to use pykeen.datasets.{self.__class__.__name__}.',
            ) from e

        dataset = LinkPropPredDataset(name=self.name, root=self.cache_root)
        edge_split = dataset.get_edge_split()
        self._training = self._make_tf(edge_split["train"])
        assert self._training is not None  # makes mypy hapy
        self._testing = self._make_tf(
            edge_split["test"],
            entity_to_id=self._training.entity_to_id,
            relation_to_id=self._training.relation_to_id,
        )
        self._validation = self._make_tf(
            edge_split["valid"],
            entity_to_id=self._training.entity_to_id,
            relation_to_id=self._training.relation_to_id,
        )

    def _loaded_validation(self) -> bool:
        return self._loaded

    def _load_validation(self) -> None:
        pass

    def _make_tf(self, x, entity_to_id=None, relation_to_id=None):
        triples = np.stack([x['head'], x['relation'], x['tail']], axis=1)

        # FIXME these are already identifiers
        triples = triples.astype(np.str)

        return TriplesFactory.from_labeled_triples(
            triples=triples,
            create_inverse_triples=self.create_inverse_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )


@parse_docdata
class OGBBioKG(OGBLoader):
    """The OGB BioKG dataset.

    .. seealso:: https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg

    ---
    name: OGB BioKG
    citation:
        author: Hu
        year: 2020
        link: https://arxiv.org/abs/2005.00687
    statistics:
        entities: 45085
        relations: 51
        training: 4762677
        testing: 162870
        validation: 162886
        triples: 5088433
    """

    name = 'ogbl-biokg'

    def _make_tf(self, x, entity_to_id=None, relation_to_id=None):
        head_triples = _array(x, 'head_type', 'head')
        tail_triples = _array(x, 'tail_type', 'tail')
        triples = np.stack([head_triples, x['relation'], tail_triples], axis=1).astype(np.str)

        return TriplesFactory.from_labeled_triples(
            triples=triples,
            create_inverse_triples=self.create_inverse_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )


def _array(df, entity_type_label, entity_label):
    return np.array(
        [f'{entity_type}:{entity}' for entity_type, entity in zip(df[entity_type_label], df[entity_label])],
        dtype=np.str,
    )


@parse_docdata
class OGBWikiKG(OGBLoader):
    """The OGB WikiKG dataset.

    .. seealso:: https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg

    ---
    name: OGB WikiKG
    citation:
        author: Hu
        year: 2020
        link: https://arxiv.org/abs/2005.00687
        github: snap-stanford/ogb
    statistics:
        entities: 2500604
        relations: 535
        training: 16109182
        testing: 598543
        validation: 429456
        triples: 17137181
    """

    name = 'ogbl-wikikg'


@click.command()
@verbose_option
def _main():
    for _cls in [OGBBioKG, OGBWikiKG]:
        _cls().summarize()


if __name__ == '__main__':
    _main()
