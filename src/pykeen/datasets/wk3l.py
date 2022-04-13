# -*- coding: utf-8 -*-

"""The Wk3l-15k dataset family.

Get a summary with ``python -m pykeen.datasets.wk3l``
"""

import logging
import pathlib
import zipfile
from abc import ABC, abstractmethod
from typing import ClassVar, Iterable, Mapping, Optional, Tuple, cast

import click
import pandas
import torch
from class_resolver import ClassResolver, HintOrType, OptionalKwargs
from docdata import parse_docdata
from more_click import verbose_option
from pystow.utils import download_from_google

from .base import LazyDataset
from ..triples import TriplesFactory
from ..typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, TorchRandomHint
from ..utils import format_relative_comparison

__all__ = [
    "MTransEDataset",
    "WK3l15k",
    "CN3l",
    "WK3l120k",
]

logger = logging.getLogger(__name__)

GOOGLE_DRIVE_ID = "1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z"
GRAPH_PAIRS = ("en_fr", "en_de")


class GraphPairCombinator:
    """A base class for combination of a graph pair into a single graph."""

    @abstractmethod
    def __call__(
        self,
        left: TriplesFactory,
        right: TriplesFactory,
        alignment: pandas.DataFrame,
        **kwargs,
    ) -> TriplesFactory:
        """
        Combine two graphs using the alignment information.

        :param left:
            the triples of the left graph
        :param right:
            the triples of the right graph
        :param alignment: columns: LEFT | RIGHT
            the alignment, i.e., pairs of matching entities
        :param kwargs:
            additional keyword-based parameters passed to :meth:`TriplesFactory.__init__`

        :return:
            a single triples factory comprising the joint graph.
        """
        raise NotImplementedError


class CollapseGraphCombinator(GraphPairCombinator):
    """This combinator merges all matching entity pairs into a single ID."""

    def __call__(
        self,
        left: TriplesFactory,
        right: TriplesFactory,
        alignment: pandas.DataFrame,
        **kwargs,
    ) -> TriplesFactory:  # noqa: D102
        raise NotImplementedError


class ExtraRelationGraphCombinator(GraphPairCombinator):
    """This combinator keeps all entities, but introduces a novel alignment relation."""

    def __call__(
        self,
        left: TriplesFactory,
        right: TriplesFactory,
        alignment: pandas.DataFrame,
        **kwargs,
    ) -> TriplesFactory:  # noqa: D102
        mapped_triples = []
        entity_to_id = {}
        relation_to_id = {}
        entity_offset = relation_offset = 0
        entity_offsets = []
        for side, tf in ((0, left), (1, right)):
            mapped_triples.append(
                tf.mapped_triples + torch.as_tensor(data=[entity_offset, relation_offset, entity_offset]).view(1, 3)
            )
            entity_to_id.update((f"{side}:{key}", value + entity_offset) for key, value in tf.entity_to_id.items())
            relation_to_id.update(
                (f"{side}:{key}", value + relation_offset) for key, value in tf.relation_to_id.items()
            )
            entity_offsets.append(entity_offset)
            entity_offset += tf.num_entities
            relation_offset += tf.num_relations

        # extra alignment relation
        relation_to_id["same-as"] = relation_offset
        # filter alignment
        mask = ~(alignment["left"].isin(left.entity_to_id) & alignment["right"].isin(right.entity_to_id))
        if mask.any():
            logger.warning(
                f"Dropping {format_relative_comparison(part=mask.sum(), total=alignment.shape[0])} alignments due to unknown labels."
            )
            alignment = alignment.loc[~mask]
        # map alignment to (new) IDs
        left_id = alignment["left"].apply(left.entity_to_id.__getitem__) + entity_offsets[0]  # offset should be zero
        right_id = alignment["right"].apply(right.entity_to_id.__getitem__) + entity_offsets[1]
        # append alignment triples
        mapped_triples.append(
            torch.stack(
                [
                    torch.as_tensor(left_id, dtype=torch.long),
                    torch.full(size=(len(left_id),), fill_value=relation_offset),
                    torch.as_tensor(right_id, dtype=torch.long),
                ],
                dim=-1,
            )
        )
        # merged factory
        return TriplesFactory(
            mapped_triples=torch.cat(mapped_triples, dim=0),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            **kwargs,
        )


graph_combinator_resolver: ClassResolver[GraphPairCombinator] = ClassResolver.from_subclasses(
    base=GraphPairCombinator,
    default=ExtraRelationGraphCombinator,
)


class MTransEDataset(LazyDataset, ABC):
    """Base class for WK3l datasets (WK3l-15k, WK3l-120k, CN3l)."""

    #: The mapping from (graph-pair, side) to triple file name
    FILE_NAMES: ClassVar[Mapping[Tuple[str, str], str]]

    #: The internal dataset name
    DATASET_NAME: ClassVar[str]

    #: The hex digest for the zip file
    SHA512: str = (
        "b5b64db8acec2ef83a418008e8ff6ddcd3ea1db95a0a158825ea9cffa5a3c34a"
        "9aba6945674304f8623ab21c7248fed900028e71ad602883a307364b6e3681dc"
    )

    def __init__(
        self,
        graph_pair: str = "en_de",
        side: Optional[str] = "en",
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        force: bool = False,
        combination: HintOrType[GraphPairCombinator] = None,
        combination_kwargs: OptionalKwargs = None,
    ):
        """
        Initialize the dataset.

        :param graph_pair:
            The graph-pair within the dataset family (cf. GRAPH_PAIRS).
        :param side:
            The side of the graph-pair, a substring of the graph-pair selection, or None
            to get a union of both graphs with a special alignment relation.
        :param cache_root:
            The cache root.
        :param eager:
            Whether to directly load the dataset, or defer it to the first access of a relevant attribute.
        :param create_inverse_triples:
            Whether to create inverse triples.
        :param random_state:
            The random state used for splitting.
        :param split_ratios:
            The split ratios used for splitting the dataset into train / validation / test.
        :param force:
            Whether to enforce re-download of existing files.
        :param combination:
            the combination method to use if both sides are to be used

        :raises ValueError:
            If the graph pair or side is invalid.
        """
        # Input validation.
        if graph_pair not in GRAPH_PAIRS:
            raise ValueError(f"Invalid graph pair: Allowed are: {GRAPH_PAIRS}")
        available_sides = graph_pair.split("_")
        if not (side is None or side in available_sides):
            raise ValueError(f"side must be one of {available_sides} or None")
        self.side = side
        self.graph_pair = graph_pair
        self.combination = graph_combinator_resolver.make(combination, combination_kwargs) if side is None else None

        # For downloading
        self.drive_id = GOOGLE_DRIVE_ID
        self.force = force
        self.cache_root = self._help_cache(cache_root)

        # For splitting
        self.random_state = random_state
        self.ratios = split_ratios

        # Whether to create inverse triples
        self.create_inverse_triples = create_inverse_triples

        if eager:
            self._load()

    def _extend_cache_root(self, cache_root: pathlib.Path) -> pathlib.Path:  # noqa: D102
        # shared directory for multiple datasets.
        return cache_root.joinpath("wk3l")

    @classmethod
    def _relative_path(cls, graph_pair: str, key: Optional[str]) -> pathlib.PurePath:
        return pathlib.PurePosixPath(
            "data",
            cls.DATASET_NAME,
            graph_pair,
            cls.FILE_NAMES[graph_pair, key],
        )

    @classmethod
    def _load_graph(cls, zip_path: pathlib.Path, graph_pair: str, side: str, **kwargs) -> TriplesFactory:
        relative_path = cls._relative_path(graph_pair=graph_pair, key=side)
        # read all triples from file
        with zipfile.ZipFile(zip_path) as zf:
            logger.info(f"Reading from {zip_path} : {relative_path}")
            with zf.open(str(relative_path), mode="r") as triples_file:
                df = pandas.read_csv(
                    triples_file,
                    delimiter="@@@",
                    header=None,
                    names=[LABEL_HEAD, LABEL_RELATION, LABEL_TAIL],
                    engine="python",
                    encoding="utf8",
                )
        # some "entities" have numeric labels
        # pandas.read_csv(..., dtype=str) does not work properly.
        df = df.astype(dtype=str)

        # create triples factory
        return TriplesFactory.from_labeled_triples(
            triples=df.values,
            metadata=dict(path=zip_path, graph_pair=graph_pair, side=side),
            **kwargs,
        )

    @classmethod
    def _load_alignment(cls, zip_path: pathlib.Path, graph_pair: str) -> pandas.DataFrame:
        """Load entity alignment information for the given graph pair."""
        left, right = graph_pair.split("_")
        dfs = []
        for key, names in ((f"{left}->{right}", ["left", "right"]), (f"{right}->{left}", ["right", "left"])):
            relative_path = cls._relative_path(graph_pair=graph_pair, key=key)
            with zipfile.ZipFile(zip_path) as zf:
                logger.info(f"Reading from {zip_path} : {relative_path}")
                with zf.open(str(relative_path), mode="r") as file:
                    df = pandas.read_csv(
                        file,
                        delimiter="@@@",
                        header=None,
                        names=names,
                        engine="python",
                        encoding="utf8",
                    )
            # some "entities" have numeric labels
            # pandas.read_csv(..., dtype=str) does not work properly.
            df = df.astype(dtype=str)
            dfs.append(df)
        return pandas.concat(dfs)

    def _load(self) -> None:
        path = self.cache_root.joinpath("data.zip")

        # ensure file is present
        # TODO: Re-use ensure_from_google?
        if not path.is_file() or self.force:
            logger.info(f"Downloading file from Google Drive (ID: {self.drive_id})")
            download_from_google(self.drive_id, path, hexdigests=dict(sha512=self.SHA512))

        if self.side is None:
            assert self.combination is not None
            left_side, right_side = self.graph_pair.split("_")
            tf = self.combination(
                left=self._load_graph(
                    zip_path=path,
                    graph_pair=self.graph_pair,
                    side=left_side,
                ),
                right=self._load_graph(
                    zip_path=path,
                    graph_pair=self.graph_pair,
                    side=right_side,
                ),
                alignment=self._load_alignment(
                    zip_path=path,
                    graph_pair=self.graph_pair,
                ),
            )
        else:
            # create triples factory
            tf = self._load_graph(
                zip_path=path,
                graph_pair=self.graph_pair,
                side=self.side,
                create_inverse_triples=self.create_inverse_triples,
            )

        # split
        self._training, self._testing, self._validation = cast(
            Tuple[TriplesFactory, TriplesFactory, TriplesFactory],
            tf.split(
                ratios=self.ratios,
                random_state=self.random_state,
            ),
        )
        logger.info("[%s] done splitting data from %s", self.__class__.__name__, path)

    def _load_validation(self) -> None:
        pass  # already loaded by _load()

    def _extra_repr(self) -> Iterable[str]:
        yield from super()._extra_repr()
        yield f"self.graph_pair={self.graph_pair}"
        yield f"self.side={self.side}"


@parse_docdata
class WK3l15k(MTransEDataset):
    """The WK3l-15k dataset family.

    .. note ::
        This dataset contains artifacts from incorrectly treating literals as entities.

    ---
    name: WK3l-15k Family
    citation:
        author: Chen
        year: 2017
        link: https://www.ijcai.org/Proceedings/2017/0209.pdf
    single: true
    statistics:
        entities: 15126
        relations: 1841
        triples: 209041
        training: 167232
        testing: 20904
        validation: 20905
    """

    DATASET_NAME = "WK3l-15k"
    FILE_NAMES = {
        ("en_de", "en"): "P_en_v6.csv",
        ("en_de", "de"): "P_de_v6.csv",
        ("en_de", "en->de"): "en2de_fk.csv",  # left-to-right entity alignment
        ("en_de", "de->en"): "de2en_fk.csv",  # right-to-left entity alignment
        ("en_de", None): "P_en_de_v6.csv",  # triple alignment
        ("en_fr", "en"): "P_en_v5.csv",
        ("en_fr", "fr"): "P_fr_v5.csv",
        ("en_fr", "en->fr"): "en2fr_fk.csv",  # left-to-right entity alignment
        ("en_fr", "fr->en"): "fr2en_fk.csv",  # right-to-left entity alignment
        ("en_fr", None): "P_en_fr_v5.csv",  # triple alignment
    }


@parse_docdata
class WK3l120k(MTransEDataset):
    """The WK3l-120k dataset family.

    .. note ::
        This dataset contains artifacts from incorrectly treating literals as entities.

    ---
    name: WK3l-120k Family
    citation:
        author: Chen
        year: 2017
        link: https://www.ijcai.org/Proceedings/2017/0209.pdf
    single: true
    statistics:
        entities: 119748
        relations: 3109
        triples: 1375406
        training: 499727
        testing: 62466
        validation: 62466
    """

    DATASET_NAME = "WK3l-120k"
    FILE_NAMES = {
        ("en_de", "en"): "P_en_v6_120k.csv",
        ("en_de", "de"): "P_de_v6_120k.csv",
        ("en_fr", "en"): "P_en_v5_120k.csv",
        ("en_fr", "fr"): "P_fr_v5_120k.csv",
    }


@parse_docdata
class CN3l(MTransEDataset):
    """The CN3l dataset family.

    ---
    name: CN3l Family
    citation:
        author: Chen
        year: 2017
        link: https://www.ijcai.org/Proceedings/2017/0209.pdf
    single: true
    statistics:
        entities: 3206
        relations: 42
        triples: 21777
        training: 23492
        testing: 2936
        validation: 2937
    """

    DATASET_NAME = "CN3l"
    FILE_NAMES = {
        ("en_de", "en"): "C_en_d.csv",
        ("en_de", "de"): "C_de.csv",
        ("en_fr", "en"): "C_en_f.csv",
        ("en_fr", "fr"): "C_fr.csv",
    }


@click.command()
@verbose_option
def _main():
    for graph_pair in GRAPH_PAIRS:
        for side in [None] + graph_pair.split("_"):
            for cls in (WK3l15k, WK3l120k, CN3l):
                ds = cls(graph_pair=graph_pair, side=side)
                ds.summarize()


if __name__ == "__main__":
    _main()
