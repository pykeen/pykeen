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

from pykeen.datasets import LazyDataset
from pykeen.triples import TriplesFactory
from pykeen.typing import TorchRandomHint

__all__ = [
    "WK3l15k",
]

logger = logging.getLogger(__name__)

GOOGLE_DRIVE_ID = "1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z"
GRAPH_PAIRS = ("en_fr", "en_de")
SIDES = tuple(sum((pair.split("_") for pair in GRAPH_PAIRS), start=[]))


class WK3l15k(LazyDataset):
    """The WK3l-15k dataset family.

    The datasets were first described in https://www.ijcai.org/Proceedings/2017/0209.pdf.
    """

    ratios = (0.8, 0.1, 0.1)

    def __init__(
        self,
        graph_pair: str = "en_de",
        side: str = "en",
        cache_root: Optional[str] = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = None,
        force: bool = False,
    ):
        if graph_pair not in GRAPH_PAIRS:
            raise ValueError(f"Invalid graph pair: Allowed are: {GRAPH_PAIRS}")
        available_sides = graph_pair.split("_")
        if side not in available_sides:
            raise ValueError(f"side must be one of {available_sides}")
        suffix = 5 if graph_pair == "en_fr" else 6
        file_name = f"P_{side}_v{suffix}.csv"
        self._relative_path = pathlib.PurePosixPath("data", "WK3l-15k", graph_pair, file_name)
        self.cache_root = self._help_cache(cache_root)

        self.random_state = random_state
        self.drive_id = GOOGLE_DRIVE_ID
        self.name = "wk3l15k.zip"
        self.create_inverse_triples = create_inverse_triples
        self.force = force

        if eager:
            self._load()

    def _get_path(self) -> pathlib.Path:
        return self.cache_root.joinpath(self.name)

    def _load(self) -> None:
        path = self._get_path()

        # ensure file is present
        if not path.is_file() or self.force:
            download_from_google(self.drive_id, path)

        # read all triples from file
        with zipfile.ZipFile(path) as zf:
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
        df = df.astype(dtype=str)

        # create triples factory
        tf_path = path
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
        logger.info("[%s] done splitting data from %s", self.__class__.__name__, tf_path)

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
