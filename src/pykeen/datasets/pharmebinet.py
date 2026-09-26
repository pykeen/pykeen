"""The `PharMeBINet <https://github.com/ckoenigs/PharMeBINet/>`_ dataset.

Get a summary with ``python -m pykeen.datasets.pharmebinet``.
"""

from typing import Unpack

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import CompressedSingleDatasetKwargs, TarFileSingleDataset

__all__ = [
    "PharMeBINet",
]

RAW_URL = "https://zenodo.org/record/7011027/files/pharmebinet_tsv_2022_08_19_v2.tar.gz"


@parse_docdata
class PharMeBINet(TarFileSingleDataset):
    """The PharMeBINet dataset from [koenigs2022]_.

    ---
    name: PharMeBINet
    citation:
        github: ckoenigs/PharMeBINet
        author: Königs
        year: 2022
        link: https://www.nature.com/articles/s41597-022-01510-3
    single: true
    statistics:
        entities: 2869407
        relations: 208
        triples: 15883653
        training: 12702210
        testing: 1587776
        validation: 1587777
    """

    def __init__(self, **kwargs: Unpack[CompressedSingleDatasetKwargs]) -> None:
        """Initialize the PharMeBINet dataset from [koenigs2022]_."""
        kwargs.setdefault("random_state", 0)
        kwargs["read_csv_kwargs"] = {
            "usecols": ["start_id", "type", "end_id"],
            "sep": "\t",
            "dtype": {"start_id": str, "end_id": str},
        }
        super().__init__(url=RAW_URL, relative_path="edges.tsv", **kwargs)


@click.command()
@verbose_option
def _main() -> None:
    from pykeen.datasets import get_dataset

    get_dataset(dataset=PharMeBINet).summarize()


if __name__ == "__main__":
    _main()
