# -*- coding: utf-8 -*-

"""The Open Academic Graph (OAG) dataset.

Get a summary with ``python -m pykeen.datasets.oag``
"""

import itertools as itt
import json
import zipfile
from typing import Iterable, Mapping, Tuple

import pandas as pd
import pystow
from more_itertools import roundrobin
from pystow.utils import name_from_url

from pykeen.datasets.base import TabbedDataset

__all__ = [
    'OAG',
]

VENUE_LINKING_URL = 'https://academicgraphv2.blob.core.windows.net/oag/linkage/venue_linking_pairs.zip'
AUTHOR_LINKING_URL = 'https://academicgraphv2.blob.core.windows.net/oag/linkage/author_linking_pairs.zip'


class OAG(TabbedDataset):
    """The Open Academic Graph (OAG) dataset."""

    def _get_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(itt.chain(_iterables())))


def _iterables():
    return [
        _iter_triples(VENUE_LINKING_URL, 'mid', 'aid', 'venue'),
        _iter_triples(AUTHOR_LINKING_URL, 'mid', 'aid', 'author'),
    ]


def main():
    i = 0
    for line in roundrobin(_iterables()):
        print(line)
        i += 1
        if i > 15:
            break


def _iter_triples(url: str, s_key: str, o_key: str, rel: str) -> Iterable[Tuple[str, str, str]]:
    for line in _iter_venue_linking(url):
        yield line[s_key], rel, line[o_key]


def _iter_venue_linking(url: str) -> Mapping[str, str]:
    name = name_from_url(url)
    name = f'{name[:-len(".zip")]}.txt'

    path = pystow.ensure('pykeen', 'datasets', 'oag', url=url)
    with zipfile.ZipFile(path) as zip_file:
        with zip_file.open(name) as file:
            for line in file:
                yield json.loads(line)


if __name__ == '__main__':
    main()
