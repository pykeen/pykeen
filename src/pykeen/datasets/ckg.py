# -*- coding: utf-8 -*-

"""Clinical Knowledge Graph."""

import tarfile
from pathlib import Path
from typing import Iterable

import click
import pandas as pd

import pystow
from pykeen.datasets.base import TabbedDataset

__all__ = [
    'CKG',
]

URL = 'https://data.mendeley.com/public-files/datasets/mrcf7f4tc2/files/c0d058a2-adfa-4b96-97d9-c9ec7fc5adb9/file_downloaded'


class CKG(TabbedDataset):
    """The Clinical Knowledge Graph (CKG) dataset."""

    def _get_df(self) -> pd.DataFrame:
        return _get_df()


def _get_df() -> pd.DataFrame:
    path = pystow.get('pykeen', 'datasets', 'ckg', 'preloaded.tsv.gz')
    if path.exists():
        return pd.read_csv(path, sep='\t')
    df = pd.concat(_iterate_dataframes())
    df.to_csv(path, sep='\t', index=False)
    return df


COLUMNS = ['START_ID', 'TYPE', 'END_ID']

def _iterate_dataframes() -> Iterable[pd.DataFrame]:
    archive_path = pystow.ensure('pykeen', 'datasets', 'ckg', url=URL, name='data.tar.gz')
    with tarfile.TarFile.open(archive_path) as tar_file:
        for tarinfo in tar_file:
            if not tarinfo.name.startswith('data/imports/') or not tarinfo.name.endswith('.tsv'):
                continue
            path = Path(tarinfo.name)
            if path.name.startswith('.'):
                continue
            with tar_file.extractfile(tarinfo) as file:
                df = pd.read_csv(file, usecols=COLUMNS, sep='\t', dtype=str)
                df = df[COLUMNS]
                print(path)
                print(df.head())
                yield df


@click.command()
def _main():
    d = CKG()
    d.summarize()


if __name__ == '__main__':
    _main()
