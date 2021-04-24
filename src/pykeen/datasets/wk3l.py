# -*- coding: utf-8 -*-

"""The Wk3l-15k dataset family.

Get a summary with ``python -m pykeen.datasets.wk3l``
"""
from pathlib import Path
from typing import Union

import click
from more_click import verbose_option

from pykeen.datasets.base import ZipFileSingleDataset
from pystow.utils import download_from_google

GOOGLE_DRIVE_ID = '1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z'
GRAPH_PAIRS = ("en_fr", "en_de")
SIDES = tuple(sum((pair.split("_") for pair in GRAPH_PAIRS), start=[]))


class WK3l15k(ZipFileSingleDataset):
    """The WK3l-15k dataset family.

    The datasets were first described in https://www.ijcai.org/Proceedings/2017/0209.pdf.
    """

    def __init__(
        self,
        graph_pair: str = "en_de",
        side: str = "en",
        **kwargs,
    ):
        if graph_pair not in GRAPH_PAIRS:
            raise ValueError(f"Invalid graph pair: Allowed are: {GRAPH_PAIRS}")
        if side not in graph_pair.split("_"):
            raise ValueError(f"side must be one of {graph_pair.split('_')}")
        suffix = 5 if graph_pair == "en_fr" else 6
        file_name = f"P_{side}_v{suffix}.csv"
        super().__init__(
            url=GOOGLE_DRIVE_ID,
            relative_path=f"data/WK3l-15k/{graph_pair}/{file_name}",
            name="wk3l15k.zip",
            **kwargs,
        )

    @staticmethod
    def _download(location: str, path: Union[str, Path]):
        download_from_google(location, path)


@click.command()
@click.option("--graph-pair", type=click.Choice(GRAPH_PAIRS, case_sensitive=True), default="en_de")
@click.option("--side", type=click.Choice(SIDES, case_sensitive=True), default="en")
@verbose_option
def _main(
    graph_pair: str,
    side: str,
):
    ds = WK3l15k(graph_pair=graph_pair, side=side)
    ds.summarize()


if __name__ == '__main__':
    _main()
