# -*- coding: utf-8 -*-

"""The `Common Sense Knowledge Graph <https://github.com/usc-isi-i2/cskg>`_ dataset.

- GitHub Repository: https://github.com/usc-isi-i2/cskg
- Paper: https://arxiv.org/pdf/2012.11490.pdf
- Data download: https://zenodo.org/record/4331372/files/cskg.tsv.gz
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
