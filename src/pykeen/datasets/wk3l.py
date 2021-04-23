# -*- coding: utf-8 -*-

"""The Wk3l-15k dataset family.

Get a summary with ``python -m pykeen.datasets.WK3l15k``
"""

import click
from more_click import verbose_option

from pykeen.datasets.base import ZipFileSingleDataset

# TODO: Download does not work properly for Google Drive
URL = 'https://drive.google.com/open?id=1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z'

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
            url=URL,
            relative_path=f"data/WK3l-15k/{graph_pair}/{file_name}",
            name="wk3l15k.zip",
            **kwargs,
        )


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
