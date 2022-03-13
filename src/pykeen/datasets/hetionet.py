# -*- coding: utf-8 -*-

"""The Hetionet dataset.

Get a summary with ``python -m pykeen.datasets.hetionet``
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import SingleTabbedDataset
from ..typing import TorchRandomHint

__all__ = [
    "Hetionet",
]

URL = "https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz"


@parse_docdata
class Hetionet(SingleTabbedDataset):
    """The Hetionet dataset from [himmelstein2017]_.

    In its publication [himmelstein2017]_, it is demonstrated to be useful for link prediction in drug repositioning
    and made publicly available through its `GitHub repository <https://github.com/hetio/hetionet>`_ in several formats.
    The link prediction algorithm showcased does not rely on embeddings, which leaves room for interesting comparison.
    One such comparison was made during the master's thesis of Lingling Xu [xu2019]_.
    ---
    name: Hetionet
    citation:
        author: Himmelstein
        year: 2017
        link: https://doi.org/10.7554/eLife.26726
        github: hetio/hetionet
    single: true
    statistics:
        entities: 45158
        relations: 24
        triples: 2250197
        training: 1800157
        testing: 225020
        validation: 225020
    """

    def __init__(
        self,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        **kwargs,
    ):
        """Initialize the `Hetionet <https://github.com/hetio/hetionet>`_ dataset from [himmelstein2017]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: The random seed to use in splitting the dataset. Defaults to 0.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.SingleTabbedDataset`.
        """
        super().__init__(
            url=URL,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=Hetionet)
    ds.summarize()


if __name__ == "__main__":
    _main()
