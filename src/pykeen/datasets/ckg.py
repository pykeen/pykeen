# -*- coding: utf-8 -*-

"""Clinical Knowledge Graph."""

import pathlib
import tarfile
from pathlib import Path
from typing import Iterable, Optional
from urllib.request import urlretrieve

import click
import pandas as pd
from docdata import parse_docdata
from more_click import verbose_option

from .base import TabbedDataset
from ..typing import TorchRandomHint

__all__ = [
    "CKG",
]

URL = "https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/d1e8d3df-2342-468a-91a9-97a981a479ad"
COLUMNS = ["START_ID", "TYPE", "END_ID"]


@parse_docdata
class CKG(TabbedDataset):
    """The Clinical Knowledge Graph (CKG) dataset from [santos2020]_.

    ---
    name: Clinical Knowledge Graph
    citation:
        author: Santos
        year: 2020
        link: https://doi.org/10.1101/2020.05.09.084897
        github: MannLabs/CKG
    single: true
    statistics:
        entities: 7617419
        relations: 11
        triples: 26691525
        training: 21353220
        testing: 2669152
        validation: 2669153
    """

    def __init__(
        self,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        **kwargs,
    ):
        """Initialize the `CKG <https://github.com/MannLabs/CKG>`_ dataset from [santos2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: The random seed to use in splitting the dataset. Defaults to 0.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TabbedDataset`.
        """
        super().__init__(
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            **kwargs,
        )
        self.preloaded_path = self.cache_root.joinpath("preloaded.tsv.gz")

    def _get_path(self) -> Optional[pathlib.Path]:
        return self.preloaded_path

    def _get_df(self) -> pd.DataFrame:
        if self.preloaded_path.exists():
            return pd.read_csv(self.preloaded_path, sep="\t", dtype=str)
        df = pd.concat(self._iterate_dataframes())
        df.to_csv(self.preloaded_path, sep="\t", index=False)
        return df

    def _iterate_dataframes(self) -> Iterable[pd.DataFrame]:
        archive_path = self.cache_root / "data.tar.gz"
        if not archive_path.exists():
            urlretrieve(URL, archive_path)  # noqa:S310
        with tarfile.TarFile.open(archive_path) as tar_file:
            if tar_file is None:
                raise ValueError
            for tarinfo in tar_file:
                if not tarinfo.name.startswith("data/imports/") or not tarinfo.name.endswith(".tsv"):
                    continue
                path = Path(tarinfo.name)
                if path.name.startswith("."):
                    continue

                _inner_file = tar_file.extractfile(tarinfo)
                if _inner_file is None:
                    raise ValueError(f"Unable to open inner file: {tarinfo}")
                with _inner_file as file:
                    df = pd.read_csv(file, usecols=COLUMNS, sep="\t", dtype=str)
                    df = df[COLUMNS]
                    yield df


@click.command()
@verbose_option
def _main():
    from pykeen.datasets import get_dataset

    d = get_dataset(dataset=CKG)
    d.summarize()


if __name__ == "__main__":
    _main()
