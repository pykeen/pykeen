# -*- coding: utf-8 -*-

"""The `Common Sense Knowledge Graph <https://github.com/usc-isi-i2/cskg>`_ dataset.

- GitHub Repository: https://github.com/usc-isi-i2/cskg
- Paper: https://arxiv.org/abs/2012.11490
- Data download: https://zenodo.org/record/4331372/files/cskg.tsv.gz
"""

import logging

from docdata import parse_docdata

from .base import SingleTabbedDataset
from ..typing import TorchRandomHint

__all__ = [
    "CSKG",
]

URL = "https://zenodo.org/record/4331372/files/cskg.tsv.gz"


@parse_docdata
class CSKG(SingleTabbedDataset):
    """The CSKG dataset.

    The CSKG combines several knowledge graphs with "common sense" knowledge.
    ---
    name: Commonsense Knowledge Graph
    citation:
        author: Ilievski
        year: 2020
        link: http://arxiv.org/abs/2012.11490
        github: usc-isi-i2/cskg
    single: true
    statistics:
        entities: 2087833
        relations: 58
        triples: 4598728
        training: 4598728
        testing: 574841
        validation: 574842
    """

    def __init__(self, create_inverse_triples: bool = False, random_state: TorchRandomHint = 0, **kwargs):
        """Initialize the `CSKG <https://github.com/usc-isi-i2/cskg>`_ dataset from [ilievski2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: The random seed to use in splitting the dataset. Defaults to 0.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.SingleTabbedDataset`.
        """
        super().__init__(
            url=URL,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            read_csv_kwargs=dict(
                usecols=["node1", "relation", "node2"],
            ),
            **kwargs,
        )


def _main():
    from pykeen.datasets import get_dataset

    ds = get_dataset(dataset=CSKG)
    ds.summarize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _main()
