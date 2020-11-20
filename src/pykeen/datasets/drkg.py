# -*- coding: utf-8 -*-

"""The `DRKG <https://github.com/gnn4dr/DRKG>`_ dataset.

Get a summary with ``python -m pykeen.datasets.drkg``
"""

import logging
from typing import Union

import numpy as np

from .base import TarFileSingleDataset

__all__ = [
    'DRKG',
]

URL = 'https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz'


class DRKG(TarFileSingleDataset):
    """The DRKG dataset.

    This is a medium-sized biological knowledge graph including 97,238 entities, 13 entity types,
    107 relations, and 5,874,261 triples.

    .. seealso:: https://github.com/gnn4dr/DRKG
    """

    def __init__(
        self,
        create_inverse_triples: bool = False,
        random_state: Union[None, int, np.random.RandomState] = 0,
        **kwargs,
    ):
        super().__init__(
            url=URL,
            relative_path='drkg.tsv',
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            **kwargs,
        )


def _main():
    ds = DRKG(eager=True)
    ds.summarize()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _main()
