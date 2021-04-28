# -*- coding: utf-8 -*-

"""The Wk3l-15k dataset family.

Get a summary with ``python -m pykeen.datasets.wk3l``
"""

import logging
import pathlib
import zipfile
from typing import Optional, Tuple, cast

import click
import pandas
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


class WK3l15k(LazyDataset):
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

    def __init__(
        self,
        graph_pair: str = "en_de",
        side: str = "en",
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = None,
        split_ratios: Tuple[int, int, int] = (0.8, 0.1, 0.1),
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

        # compose relative file name within the archive.
        suffix = 5 if graph_pair == "en_fr" else 6
        file_name = f"P_{side}_v{suffix}.csv"
        self._relative_path = pathlib.PurePosixPath("data", "WK3l-15k", graph_pair, file_name)

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

    def _get_path(self) -> pathlib.Path:
        return self.cache_root.joinpath("wk3l15k.zip")

    def _load(self) -> None:
        path = self._get_path()

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


@click.command()
@verbose_option
def _main():
    for graph_pair in GRAPH_PAIRS:
        for side in graph_pair.split("_"):
            ds = WK3l15k(graph_pair=graph_pair, side=side)
            ds.summarize()


if __name__ == "__main__":
    _main()
