# -*- coding: utf-8 -*-

"""Load the OGB datasets.

Run with ``python -m pykeen.datasets.ogb``
"""

from typing import ClassVar, Optional, Mapping, Iterable, Sequence, cast

import click
import torch
from docdata import parse_docdata
from more_click import verbose_option

from .base import LazyDataset
from ..triples import TriplesFactory
from ..triples.triples_factory import create_entity_mapping, create_relation_mapping
from ..typing import MappedTriples

__all__ = [
    "OGBLoader",
    "OGBBioKG",
    "OGBWikiKG2",
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
        self._create_inverse_triples = create_inverse_triples

    def _load(self) -> None:
        try:
            from ogb.linkproppred import LinkPropPredDataset
        except ImportError as e:
            raise ModuleNotFoundError(
                f"Need to `pip install ogb` to use pykeen.datasets.{self.__class__.__name__}.",
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
        # note: we do not use the built-in constants here, since those refer to OGB nomenclature
        #       (which happens to coincide with ours)
        # note: to be more memory-efficient we do not convert to numpy array of strings
        entity_to_id = entity_to_id or create_entity_mapping(heads=x["head"], tails=x["tail"])
        relation_to_id = relation_to_id or create_relation_mapping(relations=x["relation"])
        # convert to integers
        mapped_triples = cast(
            MappedTriples,
            torch.as_tensor(
                data=[
                    [entity_to_id[h], relation_to_id[r], entity_to_id[t]]
                    for h, r, t in zip(x["head"], x["relation"], x["tail"])
                ],
                dtype=torch.long,
            ),
        )
        return TriplesFactory(
            mapped_triples=mapped_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            create_inverse_triples=self.create_inverse_triples,
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
        entities: 93773
        relations: 51
        training: 4762677
        testing: 162870
        validation: 162886
        triples: 5088434
    """

    name = "ogbl-biokg"

    def _make_tf(self, x, entity_to_id=None, relation_to_id=None):
        # note: to be more memory-efficient we do not convert to numpy array of strings
        # heads and tails need to be prefixed by the type to avoid name collision
        return super()._make_tf(
            x=dict(
                relation=x["relation"],
                head=_combine_labels(x, "head_type", "head"),
                tail=_combine_labels(x, "tail_type", "tail"),
            ),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )


def _combine_labels(d: Mapping[str, Iterable[str]], entity_type_label: str, entity_label: str) -> Sequence[str]:
    return [f"{entity_type}:{entity}" for entity_type, entity in zip(d[entity_type_label], d[entity_label])]


@parse_docdata
class OGBWikiKG2(OGBLoader):
    """The OGB WikiKG2 dataset.

    .. seealso:: https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg2

    ---
    name: OGB WikiKG2
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

    name = "ogbl-wikikg2"


@click.command()
@verbose_option
def _main():
    for _cls in [OGBBioKG, OGBWikiKG2]:
        _cls().summarize()


if __name__ == "__main__":
    _main()
