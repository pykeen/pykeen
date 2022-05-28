# -*- coding: utf-8 -*-

"""The OpenEA dataset family.

Get a summary with ``python -m pykeen.datasets.openea``
"""

import itertools
import logging
import pathlib
from typing import Literal, Tuple

import click
import pandas
from docdata import parse_docdata
from more_click import verbose_option
from pystow.utils import read_zipfile_csv

from .base import EADataset
from ...constants import PYKEEN_DATASETS_MODULE
from ...triples import TriplesFactory
from ...typing import EA_SIDE_LEFT, EA_SIDES, LABEL_HEAD, LABEL_RELATION, LABEL_TAIL, EASide

__all__ = [
    "OpenEA",
]

OPEN_EA_MODULE = PYKEEN_DATASETS_MODULE.submodule("openea")

logger = logging.getLogger(__name__)

# graph pairs
GraphPair = Literal["D_W", "D_Y", "EN_DE", "EN_FR"]
D_W: GraphPair = "D_W"
D_Y: GraphPair = "D_Y"
EN_DE: GraphPair = "EN_DE"
EN_FR: GraphPair = "EN_FR"
GRAPH_PAIRS: Tuple[GraphPair, ...] = (D_W, D_Y, EN_DE, EN_FR)

# graph sizes
GraphSize = Literal["15K", "100K"]
SIZE_15K: GraphSize = "15K"
SIZE_100K: GraphSize = "15K"
GRAPH_SIZES = (SIZE_15K, SIZE_100K)

# graph versions
GraphVersion = Literal["V1", "V2"]
V1: GraphVersion = "V1"
V2: GraphVersion = "V2"
GRAPH_VERSIONS = (V1, V2)


@parse_docdata
class OpenEA(EADataset):
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
    SHA512: str = (
        "c1589f185f86e05c497de147b4d6c243c66775cb4b50c6b41ecc71b36cfafb4c"
        "9f86fbee94e1e78a7ee056dd69df1ce3fc210ae07dc64955ad2bfda7450545ef"
    )

    def __init__(
        self,
        *,
        graph_pair: str = D_W,
        size: GraphSize = SIZE_15K,
        version: GraphVersion = V1,
        **kwargs,
    ):
        """
        Initialize the dataset.

        :param graph_pair:
            The graph-pair within the dataset family (cf. GRAPH_PAIRS).
        :param size:
            The size of the graphs (either "15K" or "100K").
        :param version:
            The version of the pairing (either "V1" or "V2). "V1" has lower connectivity in the graph than "V2".
        :param kwargs:
            additional keyword-based parameters passed to :meth:`EABase.__init__`

        :raises ValueError:
            If the graph pair, size or version is invalid.
        """
        # Input validation.
        if graph_pair not in GRAPH_PAIRS:
            raise ValueError(f"Invalid graph pair: Allowed are: {GRAPH_PAIRS}")
        if size not in GRAPH_SIZES:
            raise ValueError(f"size must be one of {GRAPH_SIZES}")
        if version not in GRAPH_VERSIONS:
            raise ValueError(f"version must be one of {GRAPH_VERSIONS}")

        # ensure zip file is present
        self.zip_path = OPEN_EA_MODULE.ensure(
            url=OpenEA.FIGSHARE_LINK,
            name="OpenEA_dataset_v2.0.zip",
            download_kwargs=dict(hexdigests=dict(sha512=OpenEA.SHA512)),
        )
        # save relative paths beforehand so they are present for loading
        self.inner_path = pathlib.PurePosixPath("OpenEA_dataset_v2.0", f"{graph_pair}_{size}_{version}")
        # delegate to super class
        super().__init__(**kwargs)

    # docstr-coverage: inherited
    def _load_graph(self, side: EASide) -> TriplesFactory:  # noqa: D102
        # left side has files ending with 1, right side with 2
        one_or_two = "1" if side == EA_SIDE_LEFT else "2"
        file_name = f"rel_triples_{one_or_two}"
        return TriplesFactory.from_labeled_triples(
            triples=read_zipfile_csv(
                path=self.zip_path,
                inner_path=str(self.inner_path.joinpath(file_name)),
                header=None,
                names=[LABEL_HEAD, LABEL_RELATION, LABEL_TAIL],
                sep="\t",
                encoding="utf8",
                dtype=str,
            ).values,
            metadata={"path": self.zip_path},
        )

    # docstr-coverage: inherited
    def _load_alignment(self) -> pandas.DataFrame:  # noqa: D102
        return read_zipfile_csv(
            path=self.zip_path,
            inner_path=str(self.inner_path.joinpath("ent_links")),
            header=None,
            names=list(EA_SIDES),
            sep="\t",
            encoding="utf8",
            dtype=str,
        )


@click.command()
@verbose_option
def _main():
    for size, version, graph_pair, side in itertools.product(
        GRAPH_SIZES, GRAPH_VERSIONS, GRAPH_PAIRS, EA_SIDES + (None,)
    ):
        ds = OpenEA(graph_pair=graph_pair, side=side, size=size, version=version)
        ds.summarize()


if __name__ == "__main__":
    _main()
