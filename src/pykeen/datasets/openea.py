# -*- coding: utf-8 -*-

"""The OpenEA dataset family.

Get a summary with ``python -m pykeen.datasets.openea``
"""

import logging
import pathlib
from typing import Optional, Tuple, cast

import click
from docdata import parse_docdata
from more_click import verbose_option
from pystow.utils import download, read_zipfile_csv

from .base import LazyDataset
from ..triples import TriplesFactory
from ..typing import LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, TorchRandomHint

__all__ = [
    "OpenEA",
]

logger = logging.getLogger(__name__)

GRAPH_PAIRS = ("D_W", "D_Y", "EN_DE", "EN_FR")
GRAPH_SIZES = ("15K", "100K")
GRAPH_VERSIONS = ("V1", "V2")


@parse_docdata
class OpenEA(LazyDataset):
    """The OpenEA dataset family.

    ---
    name: OpenEA Family
    citation:
        author: Sun
        year: 2020
        link: http://www.vldb.org/pvldb/vol13/p2326-sun.pdf
    single: true
    statistics:
        entities: 15000
        relations: 248
        triples: 38265
        training: 30612
        testing: 3826
        validation: 3827
    """

    #: The link to the zip file
    FIGSHARE_LINK: str = "https://figshare.com/ndownloader/files/34234391"

    #: The hex digest for the zip file
    SHA512: str = "c1589f185f86e05c497de147b4d6c243c66775cb4b50c6b41ecc71b36cfafb4c9f86fbee94e1e78a7ee056dd69df1ce3fc210ae07dc64955ad2bfda7450545ef"  # noqa: E501

    def __init__(
        self,
        graph_pair: str = "D_W",
        side: str = "D",
        size: str = "15K",
        version: str = "V1",
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        force: bool = False,
    ):
        """
        Initialize the dataset.

        :param graph_pair:
            The graph-pair within the dataset family (cf. GRAPH_PAIRS).
        :param side:
            The side of the graph-pair, a substring of the graph-pair selection.
        :param size:
            The size of the graphs (either "15K" or "100K").
        :param version:
            The version of the pairing (either "V1" or "V2). "V1" has lower connectivity in the graph than "V2".
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
        if size not in GRAPH_SIZES:
            raise ValueError(f"size must be one of {GRAPH_SIZES}")
        if version not in GRAPH_VERSIONS:
            raise ValueError(f"version must be one of {GRAPH_VERSIONS}")

        relative_path_base = pathlib.PurePosixPath(
            "OpenEA_dataset_v2.0",
            graph_pair + "_" + size + "_" + version,
        )
        # left side has files ending with 1, right side with 2
        one_or_two = "1" if side == available_sides[0] else "2"
        self._relative_path_relations = pathlib.PurePosixPath(relative_path_base, f"rel_triples_{one_or_two}")

        # For downloading
        self.force = force
        self.cache_root = self._help_cache(cache_root)

        # For splitting
        self.random_state = random_state
        self.ratios = split_ratios

        # Whether to create inverse triples
        self.create_inverse_triples = create_inverse_triples

        if eager:
            self._load()

    def _load(self) -> None:
        path = self.cache_root.joinpath("OpenEA_dataset_v2.0.zip")

        # ensure file is present
        if not path.is_file() or self.force:
            logger.info(f"Downloading file from Figshare (Link: {self.__class__.FIGSHARE_LINK})")
            download(url=self.__class__.FIGSHARE_LINK, path=path, hexdigests={"sha512": self.SHA512})

        df = read_zipfile_csv(
            path=path,
            inner_path=str(self._relative_path_relations),
            header=None,
            names=[LABEL_HEAD, LABEL_RELATION, LABEL_TAIL],
            sep="\t",
            encoding="utf8",
            dtype=str,
        )

        # create triples factory
        tf = TriplesFactory.from_labeled_triples(
            triples=df.values,
            create_inverse_triples=self.create_inverse_triples,
            metadata={"path": path},
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


@click.command()
@verbose_option
def _main():
    for graph_pair in GRAPH_PAIRS:
        for side in graph_pair.split("_"):
            for size in GRAPH_SIZES:
                for version in GRAPH_VERSIONS:
                    ds = OpenEA(graph_pair=graph_pair, side=side, size=size, version=version)
                    ds.summarize()


if __name__ == "__main__":
    _main()
