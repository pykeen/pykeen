# -*- coding: utf-8 -*-

"""The `ConceptNet <https://conceptnet.io/>`_ dataset.

Get a summary with ``python -m pykeen.datasets.conceptnet``
"""

import click
from docdata import parse_docdata
from more_click import verbose_option

from .base import SingleTabbedDataset
from ..typing import TorchRandomHint

URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"


@parse_docdata
class ConceptNet(SingleTabbedDataset):
    """The ConceptNet dataset from [speer2017]_.

    The dataset is structured into 5 columns (see https://github.com/commonsense/conceptnet5/wiki/Downloads#assertions):
    edge URL, relation, head, tail, metadata.

    ---
    name: ConceptNet
    citation:
        author: Speer
        year: 2017
        link: https://arxiv.org/abs/1612.03975
        github: commonsense/conceptnet5
    single: true
    statistics:
        entities: 28370083
        relations: 50
        triples: 34074917
        training: 27259933
        testing: 3407492
        validation: 3407492
    """

    def __init__(
        self,
        random_state: TorchRandomHint = 0,
        **kwargs,
    ):
        """Initialize the `ConceptNet <https://github.com/commonsense/conceptnet5>`_ dataset from [speer2017]_.

        :param random_state: The random seed to use in splitting the dataset. Defaults to 0.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.SingleTabbedDataset`.
        """
        super().__init__(
            url=URL,
            random_state=random_state,
            read_csv_kwargs=dict(
                usecols=[2, 1, 3],
                header=None,
            ),
            **kwargs,
        )


@click.command()
@verbose_option
def _main():
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=ConceptNet)
    ds.summarize()


if __name__ == "__main__":
    _main()
