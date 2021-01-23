# -*- coding: utf-8 -*-

"""The `Common Sense Knowledge Graph <https://github.com/usc-isi-i2/cskg>`_ dataset.

Get a summary with ``python -m pykeen.datasets.cskg``
"""

import logging

from .base import SingleTabbedDataset
from ..typing import TorchRandomHint

__all__ = [
    'CSKG',
]

URL = 'https://zenodo.org/record/4331372/files/cskg.tsv.gz'


class CSKG(SingleTabbedDataset):
    """The CSKG dataset.

    The CSKG combines several knowledge graphs with "common sense" knowledge. It contains
    2,087,833 entities, 58 relations, and 5,748,411 triples.

    .. [ilievski2020] Ilievski, F., Szekely, P., & Zhang, B. (2020). `CSKG: The CommonSense Knowledge
       Graph <http://arxiv.org/abs/2012.11490>`_. *arxiv*, 2012.11490.
    """

    def __init__(
        self,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        eager: bool = False,
        **kwargs,
    ):
        super().__init__(
            url=URL,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            read_csv_kwargs=dict(
                usecols=['node1', 'relation', 'node2'],
            ),
            **kwargs,
        )


def _main():
    ds = CSKG(eager=True)
    ds.summarize()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _main()
