# -*- coding: utf-8 -*-

"""Clinical Knowledge Graph."""

import tarfile
from pathlib import Path
from typing import Iterable, Optional
from urllib.request import urlretrieve

import click
import pandas as pd

from .base import TabbedDataset
from ..typing import TorchRandomHint

__all__ = [
    'CKG',
]

URL = 'https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/d1e8d3df-2342-468a-91a9-97a981a479ad'
COLUMNS = ['START_ID', 'TYPE', 'END_ID']


class CKG(TabbedDataset):
    """The Clinical Knowledge Graph (CKG) dataset from [santos2020]_.

    This dataset contains ~7.6 million nodes, 11 relations, and ~26 million triples.

    .. [santos2020] Santos, A., *et al* (2020). `Clinical Knowledge Graph Integrates Proteomics Data into Clinical
       Decision-Making <https://doi.org/10.1101/2020.05.09.084897>`_. *bioRxiv*, 2020.05.09.084897.
    """

    def __init__(
        self,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        cache_root: Optional[str] = None,
    ):
        super().__init__(
            eager=eager,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            cache_root=cache_root,
        )
        self.preloaded_path = self.cache_root / 'preloaded.tsv.gz'

    def _get_path(self) -> Optional[str]:
        return self.preloaded_path.as_posix()

    def _get_df(self) -> pd.DataFrame:
        if self.preloaded_path.exists():
            return pd.read_csv(self.preloaded_path, sep='\t')
        df = pd.concat(self._iterate_dataframes())
        df.to_csv(self.preloaded_path, sep='\t', index=False)
        return df

    def _iterate_dataframes(self) -> Iterable[pd.DataFrame]:
        archive_path = self.cache_root / 'data.tar.gz'
        if not archive_path.exists():
            urlretrieve(URL, archive_path)  # noqa:S310
        with tarfile.TarFile.open(archive_path) as tar_file:
            if tar_file is None:
                raise ValueError
            for tarinfo in tar_file:
                if not tarinfo.name.startswith('data/imports/') or not tarinfo.name.endswith('.tsv'):
                    continue
                path = Path(tarinfo.name)
                if path.name.startswith('.'):
                    continue

                _inner_file = tar_file.extractfile(tarinfo)
                if _inner_file is None:
                    raise ValueError(f'Unable to open inner file: {tarinfo}')
                with _inner_file as file:
                    df = pd.read_csv(file, usecols=COLUMNS, sep='\t', dtype=str)
                    df = df[COLUMNS]
                    yield df


@click.command()
def _main():
    d = CKG()
    d.summarize()


if __name__ == '__main__':
    _main()
