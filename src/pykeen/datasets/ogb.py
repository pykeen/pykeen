# -*- coding: utf-8 -*-

"""Load the OGB datasets.

Run with ``python -m pykeen.datasets.ogb``
"""
from __future__ import annotations

import abc
import logging
import pathlib
import typing
from typing import ClassVar, Generic, Literal, Optional, Sequence, TypedDict, TypeVar, Union, cast, overload

import click
import numpy
import pandas
import torch
from docdata import parse_docdata
from more_click import verbose_option

from .base import LazyDataset
from ..triples import TriplesFactory
from ..typing import EntityMapping, RelationMapping

if typing.TYPE_CHECKING:
    from ogb.linkproppred import LinkPropPredDataset

__all__ = [
    "OGBLoader",
    "OGBBioKG",
    "OGBWikiKG2",
]

LOGGER = logging.getLogger(__name__)

# Type annotation for split types
TrainKey = Literal["train"]
EvalKey = Literal["valid", "test"]
SplitKey = Union[TrainKey, EvalKey]

# type variables for dictionaries of preprocessed data loaded through torch.load
PreprocessedTrainDictType = TypeVar("PreprocessedTrainDictType")
PreprocessedEvalDictType = TypeVar("PreprocessedEvalDictType")


class OGBLoader(LazyDataset, Generic[PreprocessedTrainDictType, PreprocessedEvalDictType]):
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

    # docstr-coverage: inherited
    def _load(self) -> None:  # noqa: D102
        dataset = self._load_ogb_dataset()
        # label mapping is in dataset.root/mapping
        entity_to_id, relation_to_id = self._load_mappings(pathlib.Path(dataset.root).joinpath("mapping"))
        self._training = TriplesFactory(
            mapped_triples=self._compose_mapped_triples(data_dict=self._load_data_dict_for_split(dataset, "train")),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            create_inverse_triples=self._create_inverse_triples,
        )
        self._testing = TriplesFactory(
            mapped_triples=self._compose_mapped_triples(data_dict=self._load_data_dict_for_split(dataset, "test")),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

    # docstr-coverage: inherited
    def _load_validation(self) -> None:  # noqa: D102
        dataset = self._load_ogb_dataset()
        self._validation = TriplesFactory(
            mapped_triples=self._compose_mapped_triples(data_dict=self._load_data_dict_for_split(dataset, "valid")),
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
        )

    def _load_ogb_dataset(self) -> "LinkPropPredDataset":
        """
        Load the OGB dataset (lazily).

        Also handles error message if OGB package is not yet installed.

        :return:
            the OGB dataset instance.

        :raises ModuleNotFoundError:
            if the OGB package is not installed
        """
        try:
            from ogb.linkproppred import LinkPropPredDataset
        except ImportError as e:
            raise ModuleNotFoundError(
                f"Need to `pip install ogb` to use pykeen.datasets.{self.__class__.__name__}.",
            ) from e
        return LinkPropPredDataset(name=self.name, root=self.cache_root)

    @overload
    def _load_data_dict_for_split(self, dataset: "LinkPropPredDataset", which: TrainKey) -> PreprocessedTrainDictType:
        ...

    @overload
    def _load_data_dict_for_split(self, dataset: "LinkPropPredDataset", which: EvalKey) -> PreprocessedEvalDictType:
        ...

    @abc.abstractmethod
    def _load_data_dict_for_split(self, dataset, which):
        """Load the dictionary of preprocessed data for the given key."""
        raise NotImplementedError

    @abc.abstractmethod
    def _load_mappings(self, mapping_root: pathlib.Path) -> tuple[EntityMapping, RelationMapping]:
        """Load entity and relation labels from the mapping root."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compose_mapped_triples(self, data_dict: PreprocessedTrainDictType | PreprocessedEvalDictType) -> numpy.ndarray:
        """Compose the mapped triples tensor for the given dataset and split."""
        raise NotImplementedError


class WikiKG2TrainDict(TypedDict):
    """A type hint for dictionaries of OGB preprocessed training triples for WikiKG2."""

    # note: we do not use the built-in constants here, since those refer to OGB nomenclature
    #       (which happens to coincide with ours)

    # dtype: numpy.int64, shape: (m,)
    head: numpy.ndarray
    relation: numpy.ndarray
    tail: numpy.ndarray


class WikiKG2EvalDict(WikiKG2TrainDict):
    """A type hint for dictionaries of OGB preprocessed evaluation triples for WikiKG2."""

    # dtype: numpy.int64, shape: (n, k)
    head_neg: numpy.ndarray
    tail_neg: numpy.ndarray


@parse_docdata
class OGBWikiKG2(OGBLoader[WikiKG2TrainDict, WikiKG2EvalDict]):
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

    # docstr-coverage: inherited
    def _load_mappings(self, mapping_root: pathlib.Path) -> tuple[EntityMapping, RelationMapping]:  # noqa: D102
        df_ent = pandas.read_csv(mapping_root.joinpath("nodeidx2entityid.csv.gz"))
        entity_to_id = dict(zip(df_ent["entity id"].tolist(), df_ent["node idx"].tolist()))
        df_rel = pandas.read_csv(mapping_root.joinpath("reltype2relid.csv.gz"))
        relation_to_id = dict(zip(df_rel["rel id"].tolist(), df_rel["reltype"].tolist()))
        return entity_to_id, relation_to_id

    # docstr-coverage: inherited
    def _load_data_dict_for_split(self, dataset, which):
        # noqa: D102
        data_dict = torch.load(
            pathlib.Path(dataset.root).joinpath("split", dataset.meta_info["split"], which).with_suffix(".pt")
        )
        if which == "train":
            data_dict = cast(WikiKG2TrainDict, data_dict)
        else:
            data_dict = cast(WikiKG2EvalDict, data_dict)
        return data_dict

    # docstr-coverage: inherited
    def _compose_mapped_triples(self, data_dict: WikiKG2TrainDict | WikiKG2EvalDict) -> numpy.ndarray:  # noqa: D102
        return numpy.stack([data_dict["head"], data_dict["relation"], data_dict["tail"]], axis=-1)


#: the node types
OGBBioKGNodeType = Literal["disease", "drug", "function", "protein", "sideeffect"]
NODE_TYPES = typing.get_args(OGBBioKGNodeType)


class BioKGTrainDict(WikiKG2TrainDict):
    """A type hint for dictionaries of OGB preprocessed training triples for BioKG."""

    # shape: (n,)
    head_type: Sequence[OGBBioKGNodeType]
    tail_type: Sequence[OGBBioKGNodeType]


class BioKGEvalDict(BioKGTrainDict):
    """A type hint for dictionaries of OGB preprocessed evaluation triples for BioKG."""

    # dtype: numpy.int64, shape: (n, k)
    head_neg: numpy.ndarray
    tail_neg: numpy.ndarray


def load_partial_entity_mapping(mapping_root: pathlib.Path, node_type: OGBBioKGNodeType) -> pandas.DataFrame:
    """Load a partial entity mapping for a single node type."""
    # disease: UMLS CUI (https://www.nlm.nih.gov/research/umls/index.html).
    # drug: STITCH ID (http://stitch.embl.de/).
    # function: Gene Ontology ID (http://geneontology.org/).
    # protein: Proteins: Entrez Gene ID (https://www.genenames.org/).
    # side effects: UMLS CUI (https://www.nlm.nih.gov/research/umls/index.html).
    # todo(@cthoyt): proper prefixing?
    df = pandas.read_csv(mapping_root.joinpath(f"{node_type}_entidx2name.csv.gz"))
    df = df.rename(columns={"ent name": "entity_name", "ent idx": "local_entity_id"})
    df["entity_type"] = node_type
    return df


@parse_docdata
class OGBBioKG(OGBLoader[BioKGTrainDict, BioKGEvalDict]):
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

    # docstr-coverage: inherited
    def _load_mappings(self, mapping_root: pathlib.Path) -> tuple[EntityMapping, RelationMapping]:  # noqa: D102
        df_rel = pandas.read_csv(mapping_root.joinpath("relidx2relname.csv.gz"))
        LOGGER.info(f"Loaded relation mapping for {len(df_rel)} relations.")
        relation_to_id = dict(zip(df_rel["rel name"].tolist(), df_rel["rel idx"].tolist()))

        # entity mappings are separate for each node type -> combine
        entity_mapping_df = pandas.concat(
            [load_partial_entity_mapping(mapping_root=mapping_root, node_type=node_type) for node_type in NODE_TYPES],
            ignore_index=True,
        ).sort_values(by=["entity_type", "entity_name"])
        entity_mapping_df["name"] = entity_mapping_df["entity_type"] + ":" + entity_mapping_df["entity_name"]
        # convert entity_name to categorical for fast joins
        entity_mapping_df["entity_type"] = entity_mapping_df["entity_type"].astype("category")
        entity_mapping_df = entity_mapping_df.reset_index(drop=False)
        LOGGER.info(f"Merged entity labels for {len(entity_mapping_df)} entities across {len(NODE_TYPES)} node types.")

        # we need the entity dataframe for fast re-mapping later on
        self.df_ent = entity_mapping_df[["index", "local_entity_id", "entity_type"]]

        entity_to_id = dict(zip(entity_mapping_df["name"].tolist(), entity_mapping_df["index"].tolist()))

        return entity_to_id, relation_to_id

    # docstr-coverage: inherited
    def _compose_mapped_triples(self, data_dict: BioKGTrainDict | BioKGEvalDict) -> numpy.ndarray:
        return numpy.stack(
            [
                self._map_entity_column(local_entity_id=data_dict["head"], entity_type=data_dict["head_type"]),
                data_dict["relation"],
                self._map_entity_column(local_entity_id=data_dict["tail"], entity_type=data_dict["tail_type"]),
            ],
            axis=-1,
        )

    # docstr-coverage: inherited
    def _load_data_dict_for_split(self, dataset, which):  # noqa: D102
        data_dict = torch.load(
            pathlib.Path(dataset.root).joinpath("split", dataset.meta_info["split"], which).with_suffix(".pt")
        )
        if which == "train":
            data_dict = cast(BioKGTrainDict, data_dict)
        else:
            data_dict = cast(BioKGEvalDict, data_dict)

        return data_dict

    def _map_entity_column(
        self, local_entity_id: numpy.ndarray, entity_type: Sequence[OGBBioKGNodeType]
    ) -> numpy.ndarray:
        """Convert node-type local entity IDs with their types to globally unique IDs."""
        # compose temporary df
        df = pandas.DataFrame({"local_entity_id": local_entity_id, "entity_type": entity_type})
        # add extra column with old index to revert sort order change by merge
        df.index.name = "old_index"
        df = df.reset_index(drop=False)
        # convert to categorical dtype
        df["entity_type"] = df["entity_type"].astype(self.df_ent["entity_type"].dtype)
        # join with entity mapping
        df = pandas.merge(df, self.df_ent, on=["local_entity_id", "entity_type"])
        assert len(df) == len(local_entity_id)
        # revert change in order
        df = df.sort_values(by="old_index")
        # select global ID
        return df["index"].values


@click.command()
@verbose_option
def _main():
    for _cls in [OGBBioKG, OGBWikiKG2]:
        _cls().summarize()


if __name__ == "__main__":
    _main()
