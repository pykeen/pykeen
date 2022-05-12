# -*- coding: utf-8 -*-

"""The Wk3l-15k dataset family.

Get a summary with ``python -m pykeen.datasets.wk3l``
"""

import itertools
import logging
import pathlib
from abc import ABC
from typing import ClassVar, Iterable, Literal, Mapping, Tuple, Union

import click
import pandas
from docdata import parse_docdata
from more_click import verbose_option
from pystow.utils import read_zipfile_csv

from .base import EADataset
from ...constants import PYKEEN_DATASETS_MODULE
from ...triples import TriplesFactory
from ...typing import EA_SIDE_LEFT, EA_SIDE_RIGHT, EA_SIDES, LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, EASide

__all__ = [
    "MTransEDataset",
    "WK3l15k",
    "CN3l",
    "WK3l120k",
]

logger = logging.getLogger(__name__)

GOOGLE_DRIVE_ID = "1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z"
GraphPair = Literal["en_de", "en_fr"]
EN_DE: GraphPair = "en_de"
EN_FR: GraphPair = "en_fr"
GRAPH_PAIRS = (EN_DE, EN_FR)
WK3L_MODULE = PYKEEN_DATASETS_MODULE.submodule("wk3l")
EA_SIDES_R: Tuple[EASide, EASide] = (EA_SIDE_RIGHT, EA_SIDE_LEFT)


class MTransEDataset(EADataset, ABC):
    """Base class for WK3l datasets (WK3l-15k, WK3l-120k, CN3l)."""

    #: The mapping from (graph-pair, side) to triple file name
    FILE_NAMES: ClassVar[Mapping[Tuple[GraphPair, Union[None, EASide, Tuple[EASide, EASide]]], str]]

    #: The internal dataset name
    DATASET_NAME: ClassVar[str]

    #: The hex digest for the zip file
    SHA512: str = (
        "b5b64db8acec2ef83a418008e8ff6ddcd3ea1db95a0a158825ea9cffa5a3c34a"
        "9aba6945674304f8623ab21c7248fed900028e71ad602883a307364b6e3681dc"
    )

    def __init__(self, graph_pair: GraphPair = EN_DE, **kwargs):
        """
        Initialize the dataset.

        :param graph_pair:
            the graph-pair within the dataset family (cf. :data:`GRAPH_PAIRS`)
        :param kwargs:
            additional keyword-based parameters passed to :meth:`EABase.__init__`

        :raises ValueError:
            if the graph pair or side is invalid
        """
        # input validation
        if graph_pair not in GRAPH_PAIRS:
            raise ValueError(f"Invalid graph pair: Allowed are: {GRAPH_PAIRS}")
        # store *before* calling super to have it available when loading the graphs
        self.graph_pair = graph_pair
        # ensure zip file is present
        self.zip_path = WK3L_MODULE.ensure_from_google(
            name="data.zip", file_id=GOOGLE_DRIVE_ID, download_kwargs=dict(hexdigests=dict(sha512=self.SHA512))
        )
        super().__init__(**kwargs)

    def _cache_sub_directories(self) -> Iterable[str]:  # noqa: D102
        # shared directory for multiple datasets.
        yield "wk3l"

    @classmethod
    def _relative_path(cls, graph_pair: GraphPair, key: Union[None, EASide, Tuple[EASide, EASide]]) -> pathlib.PurePath:
        """Determine the relative path inside the zip file."""
        return pathlib.PurePosixPath(
            "data",
            cls.DATASET_NAME,
            graph_pair,
            cls.FILE_NAMES[graph_pair, key],
        )

    def _load_df(self, key: Union[None, EASide, Tuple[EASide, EASide]], **kwargs) -> pandas.DataFrame:
        return read_zipfile_csv(
            path=self.zip_path,
            inner_path=str(self._relative_path(graph_pair=self.graph_pair, key=key)),
            header=None,
            sep="@@@",
            engine="python",
            encoding="utf8",
            dtype=str,
            keep_default_na=False,
            **kwargs,
        )

    # docstr-coverage: inherited
    def _load_graph(self, side: EASide) -> TriplesFactory:  # noqa: D102
        logger.info(f"Loading graph for side: {side}")
        df = self._load_df(key=side, names=[LABEL_HEAD, LABEL_RELATION, LABEL_TAIL])
        # create triples factory
        return TriplesFactory.from_labeled_triples(
            triples=df.values, metadata=dict(graph_pair=self.graph_pair, side=side)
        )

    # docstr-coverage: inherited
    def _load_alignment(self) -> pandas.DataFrame:  # noqa: D102
        """Load entity alignment information for the given graph pair."""
        logger.info("Loading alignment information")
        # load mappings for both sides
        dfs = [self._load_df(key=key, names=list(key)) for key in (EA_SIDES, EA_SIDES_R)]
        # load triple alignments
        df = self._load_df(
            key=None, names=[(side, column) for side in EA_SIDES for column in [LABEL_HEAD, LABEL_RELATION, LABEL_TAIL]]
        )
        # extract entity alignments
        # (h1, r1, t1) = (h2, r2, t2) => h1 = h2 and t1 = t2
        for column in [LABEL_HEAD, LABEL_TAIL]:
            part = df.loc[:, [(EA_SIDE_LEFT, column), (EA_SIDE_RIGHT, column)]].copy()
            part.columns = list(EA_SIDES)
            dfs.append(part)
        return pandas.concat(dfs)


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
        (EN_DE, EA_SIDE_LEFT): "P_en_v6.csv",
        (EN_DE, EA_SIDE_RIGHT): "P_de_v6.csv",
        (EN_DE, EA_SIDES): "en2de_fk.csv",  # left-to-right entity alignment
        (EN_DE, EA_SIDES_R): "de2en_fk.csv",  # right-to-left entity alignment
        (EN_DE, None): "P_en_de_v6.csv",  # triple alignment
        (EN_FR, EA_SIDE_LEFT): "P_en_v5.csv",
        (EN_FR, EA_SIDE_RIGHT): "P_fr_v5.csv",
        (EN_FR, EA_SIDES): "en2fr_fk.csv",  # left-to-right entity alignment
        (EN_FR, EA_SIDES_R): "fr2en_fk.csv",  # right-to-left entity alignment
        (EN_FR, None): "P_en_fr_v5.csv",  # triple alignment
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
        (EN_DE, EA_SIDE_LEFT): "P_en_v6_120k.csv",
        (EN_DE, EA_SIDE_RIGHT): "P_de_v6_120k.csv",
        (EN_DE, EA_SIDES): "en2de_fk_120k.csv",  # left-to-right entity alignment
        (EN_DE, EA_SIDES_R): "de2en_fk_120k.csv",  # right-to-left entity alignment
        (EN_DE, None): "P_en_de_v6_120k.csv",  # triple alignment
        (EN_FR, EA_SIDE_LEFT): "P_en_v5_120k.csv",
        (EN_FR, EA_SIDE_RIGHT): "P_fr_v5_120k.csv",
        (EN_FR, EA_SIDES): "en2fr_fk_120k.csv",  # left-to-right entity alignment
        (EN_FR, EA_SIDES_R): "fr2en_fk_120k.csv",  # right-to-left entity alignment
        (EN_FR, None): "P_en_fr_v5_120k.csv",  # triple alignment
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
        (EN_DE, EA_SIDE_LEFT): "C_en_d.csv",
        (EN_DE, EA_SIDE_RIGHT): "C_de.csv",
        (EN_DE, EA_SIDES): "en2de_cn.csv",  # left-to-right entity alignment
        (EN_DE, EA_SIDES_R): "de2en_cn.csv",  # right-to-left entity alignment
        (EN_DE, None): "C_en_de.csv",  # triple alignment
        (EN_FR, EA_SIDE_LEFT): "C_en_f.csv",
        (EN_FR, EA_SIDE_RIGHT): "C_fr.csv",
        (EN_FR, EA_SIDES): "en2fr_cn.csv",  # left-to-right entity alignment
        (EN_FR, EA_SIDES_R): "fr2en_cn.csv",  # right-to-left entity alignment
        (EN_FR, None): "C_en_fr.csv",  # triple alignment
    }


@click.command()
@verbose_option
def _main():
    for cls, graph_pair, side in itertools.product((WK3l15k, WK3l120k, CN3l), GRAPH_PAIRS, EA_SIDES + (None,)):
        ds = cls(graph_pair=graph_pair, side=side)
        ds.summarize()


if __name__ == "__main__":
    _main()
