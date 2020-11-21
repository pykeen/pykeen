# -*- coding: utf-8 -*-

"""Load the OGB datasets.

Run with python -m pykeen.datasets.ogb
"""

from typing import ClassVar, Optional

import numpy as np

from .base import LazyDataSet
from ..triples import TriplesFactory

__all__ = [
    'OGBLoader',
    'OGBBioKG',
    'OGBWikiKG',
]


class OGBLoader(LazyDataSet):
    """Load from the Open Graph Benchmark (OGB)."""

    #: The name of the dataset to download
    name: ClassVar[str]

    def __init__(self, cache_root: Optional[str] = None, create_inverse_triples: bool = False):
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

        return TriplesFactory(
            triples=triples,
            create_inverse_triples=self.create_inverse_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )


class OGBBioKG(OGBLoader):
    """The OGB BioKG dataset.

    .. seealso:: https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg
    """

    name = 'ogbl-biokg'


class OGBWikiKG(OGBLoader):
    """The OGB WikiKG dataset.

    .. seealso:: https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg
    """

    name = 'ogbl-wikikg'


if __name__ == '__main__':
    for _cls in [OGBBioKG, OGBWikiKG]:
        _cls().summarize()
