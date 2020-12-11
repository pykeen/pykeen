# -*- coding: utf-8 -*-

"""The Open Academic Graph (OAG) dataset.

Get a summary with ``python -m pykeen.datasets.oag``
"""

import itertools as itt
import json
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Tuple, Union

import pandas as pd
import pystow
from more_itertools import roundrobin
from pystow.utils import name_from_url

from pykeen.datasets.base import TabbedDataset

__all__ = [
    'OAG',
]

BASE_URL = 'https://academicgraphv2.blob.core.windows.net/oag'

# Links between the MAG (Microsoft Academic Graph) and AMiner (ArnetMiner) datasets
VENUE_LINKING_URL = f'{BASE_URL}/linkage/venue_linking_pairs.zip'
PAPER_LINKING_URL = f'{BASE_URL}/linkage/paper_linking_pairs.zip'
AUTHOR_LINKING_URL = f'{BASE_URL}/linkage/author_linking_pairs.zip'

# Venue downloads
AMINER_VENUES_URL = f'{BASE_URL}/aminer/venue/aminer_venues.zip'
MAG_VENUES_URL = f'{BASE_URL}/mag/venue/mag_venues.zip'

# Paper downloads
MAG_PAPERS_0_URL = f'{BASE_URL}/mag/paper/mag_papers_0.zip'
MAG_PAPERS_1_URL = f'{BASE_URL}/mag/paper/mag_papers_1.zip'
MAG_PAPERS_2_URL = f'{BASE_URL}/mag/paper/mag_papers_2.zip'
AMINER_PAPERS_0_URL = f'{BASE_URL}/aminer/paper/aminer_papers_0.zip'
AMINER_PAPERS_1_URL = f'{BASE_URL}/aminer/paper/aminer_papers_1.zip'
AMINER_PAPERS_2_URL = f'{BASE_URL}/aminer/paper/aminer_papers_2.zip'
AMINER_PAPERS_3_URL = f'{BASE_URL}/aminer/paper/aminer_papers_3.zip'

# Author downloads
MAG_AUTHORS_0_URL = f'{BASE_URL}/mag/author/mag_authors_0.zip'
MAG_AUTHORS_1_URL = f'{BASE_URL}/mag/author/mag_authors_1.zip'
MAG_AUTHORS_2_URL = f'{BASE_URL}/mag/author/mag_authors_2.zip'
AMINER_AUTHORS_0_URL = f'{BASE_URL}/aminer/author/aminer_authors_0.zip'
AMINER_AUTHORS_1_URL = f'{BASE_URL}/aminer/author/aminer_authors_1.zip'
AMINER_AUTHORS_2_URL = f'{BASE_URL}/aminer/author/aminer_authors_2.zip'
AMINER_AUTHORS_3_URL = f'{BASE_URL}/aminer/author/aminer_authors_3.zip'


class OAG(TabbedDataset):
    """The Open Academic Graph (OAG) dataset."""

    def _get_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(itt.chain(_iterables())))


def _iterables():
    return [
        _iter_triples(VENUE_LINKING_URL, 'mid', 'aid', 'sameVenue'),
        _iter_triples(AUTHOR_LINKING_URL, 'mid', 'aid', 'sameAuthor'),
    ]


def main():
    for x, y in globals().items():
        if x.endswith('_URL') and not x.startswith('BASE'):
            print(x, y)
            pystow.ensure('pykeen', 'datasets', 'oag', url=y)
    return
    i = 0
    for line in roundrobin(_iterables()):
        print(line)
        i += 1
        if i > 15:
            break


def _iter_triples(url: str, s_key: str, o_key: str, rel: str) -> Iterable[Tuple[str, str, str]]:
    for entry in _iter_zip_jsonl_linking(url):
        yield entry[s_key], rel, entry[o_key]


def _iter_zip_jsonl_linking(url: str) -> Iterable[Mapping[str, str]]:
    name = name_from_url(url)
    inner_path = f'{name[:-len(".zip")]}.txt'
    path = pystow.ensure('pykeen', 'datasets', 'oag', url=url)
    yield from _iter_zipped_jsonl(path, inner_path)


def _iter_zipped_jsonl(path: Union[str, Path], inner_path: str) -> Iterable[Mapping[str, str]]:
    with zipfile.ZipFile(path) as zip_file:
        with zip_file.open(inner_path) as file:
            for line in file:
                yield json.loads(line)


if __name__ == '__main__':
    main()
