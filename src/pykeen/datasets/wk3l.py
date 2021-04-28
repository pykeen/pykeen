# -*- coding: utf-8 -*-

"""The Wk3l-15k dataset family.

Get a summary with ``python -m pykeen.datasets.wk3l``
"""

import logging
import pathlib
import zipfile
from abc import abstractmethod
from typing import Optional, Tuple, cast

import click
import pandas
from docdata import parse_docdata
from more_click import verbose_option
from pystow.utils import download_from_google

from .base import LazyDataset
from ..triples import TriplesFactory
from ..typing import TorchRandomHint

__all__ = [
    "WK3l15k",
]

logger = logging.getLogger(__name__)

GOOGLE_DRIVE_ID = "1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z"
GRAPH_PAIRS = ("en_fr", "en_de")


class MTransEDataset(LazyDataset):
    """Base class for WK3l datasets (WK3l-15k, WK3l-120k, CN3l)."""

    def __init__(
        self,
        graph_pair: str = "en_de",
        side: str = "en",
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        force: bool = False,
    ):
        """
        Initialize the dataset.

        :param graph_pair:
            The graph-pair within the dataset family (cf. GRAPH_PAIRS).
        :param side:
            The side of the graph-pair, a substring of the graph-pair selection.
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

        :raises ValueError:
            If the graph pair or side is invalid.
        """
        # Input validation.
        if graph_pair not in GRAPH_PAIRS:
            raise ValueError(f"Invalid graph pair: Allowed are: {GRAPH_PAIRS}")
        available_sides = graph_pair.split("_")
        if side not in available_sides:
            raise ValueError(f"side must be one of {available_sides}")

        self._relative_path = self._get_relative_path(graph_pair=graph_pair, side=side)

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

    @staticmethod
    @abstractmethod
    def _get_relative_path(graph_pair: str, side: str) -> pathlib.PurePosixPath:
        """The relative path within the archive."""

    def _load(self) -> None:
        path = self.cache_root.joinpath("data.zip")

        # ensure file is present
        if not path.is_file() or self.force:
            logger.info(f"Downloading file from Google Drive (ID: {self.drive_id})")
            download_from_google(self.drive_id, path)

        # read all triples from file
        with zipfile.ZipFile(path) as zf:
            logger.info(f"Reading from {path.as_uri()}")
            with zf.open(str(self._relative_path), mode="r") as triples_file:
                df = pandas.read_csv(
                    triples_file,
                    delimiter="@@@",
                    header=None,
                    names=["head", "relation", "tail"],
                    engine="python",
                    encoding="utf8",
                )
        # some "entities" have numeric labels
        # pandas.read_csv(..., dtype=str) does not work properly.
        df = df.astype(dtype=str)

        # create triples factory
        tf = TriplesFactory.from_labeled_triples(
            triples=df.values,
            create_inverse_triples=self.create_inverse_triples,
            metadata=dict(path=path),
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


@parse_docdata
class WK3l15k(MTransEDataset):
    """The WK3l-15k dataset family.

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
    """

    @staticmethod
    def _get_relative_path(graph_pair: str, side: str) -> pathlib.PurePosixPath:  # noqa: D102
        suffix = 5 if graph_pair == "en_fr" else 6
        file_name = f"P_{side}_v{suffix}.csv"
        return pathlib.PurePosixPath("data", "WK3l-15k", graph_pair, file_name)


@parse_docdata
class WK3l120k(MTransEDataset):
    """The WK3l-120k dataset family.

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
    """

    @staticmethod
    def _get_relative_path(graph_pair: str, side: str) -> pathlib.PurePosixPath:  # noqa: D102
        suffix = 5 if graph_pair == "en_fr" else 6
        file_name = f"P_{side}_v{suffix}_120k.csv"
        return pathlib.PurePosixPath("data", "WK3l-120k", graph_pair, file_name)


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
    """
    FILE_NAMES = {
        ("en_de", "en"): "C_en_d.csv",
        ("en_de", "de"): "C_de.csv",
        ("en_fr", "en"): "C_en_f.csv",
        ("en_fr", "fr"): "C_fr.csv",
    }

    @staticmethod
    def _get_relative_path(graph_pair: str, side: str) -> pathlib.PurePosixPath:  # noqa: D102
        return pathlib.PurePosixPath("data", "CN3l", graph_pair, CN3l.FILE_NAMES[graph_pair, side])


@click.command()
@verbose_option
def _main():
    for graph_pair in GRAPH_PAIRS:
        for side in graph_pair.split("_"):
            for cls in (WK3l15k, WK3l120k, CN3l):
                ds = cls(graph_pair=graph_pair, side=side)
                ds.summarize()


if __name__ == "__main__":
    _main()
