"""Clinical Knowledge Graph."""

import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import pandas as pd
from docdata import parse_docdata
from more_click import verbose_option
from pystow.utils import DownloadKwargs, download

from .base import TabbedDataset
from .source import SimpleSource, RemoteSimpleSource
from ..typing import TorchRandomHint

__all__ = [
    "CKG",
]

URL_OLD = "https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/d1e8d3df-2342-468a-91a9-97a981a479ad"
URL = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/d1e8d3df-2342-468a-91a9-97a981a479ad"
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
        self, random_state: TorchRandomHint = 0, force: bool = False, **kwargs: Any
    ) -> None:
        """Initialize the `CKG <https://github.com/MannLabs/CKG>`_ dataset from [santos2020]_.

        :param random_state: The random seed to use in splitting the dataset. Defaults to 0.
        :param kwargs: keyword arguments passed to :class:`~pykeen.datasets.base.TabbedDataset`.
        """
        super().__init__(random_state=random_state, **kwargs)
        self.source = CKGSimpleSource(
            path=self.cache_root.joinpath("preloaded.tsv.gz"),
            raw_path=self.cache_root / "data.tar.gz",
            url=URL,
            force=force,
        )

    def _get_path(self) -> Path | None:
        return self.source.path

    def _get_df(self) -> pd.DataFrame:
        with self.source.open() as file:
            return pd.read_csv(file, sep="\t", dtype=str)


@dataclass
class CKGSimpleSource(RemoteSimpleSource):
    """A simple source for CKG."""

    raw_path: Path | None = None # TODO how to make this non-default?

    def ensure(self) -> None:
        """Download and process the CKG."""
        if self.raw_path is None:
            raise ValueError
        if self.path.is_file() and not self.force:
            return
        download(
            url=self.url,
            path=self.raw_path,
            force=self.force,
            **(self.download_kwargs or {}),
        )
        dfs: list[pd.DataFrame] = []
        with tarfile.TarFile.open(self.raw_path) as tar_file:
            if tar_file is None:
                raise ValueError
            for tarinfo in tar_file:
                if not tarinfo.name.startswith(
                    "data/imports/"
                ) or not tarinfo.name.endswith(".tsv"):
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
                    dfs.append(df)
        pd.concat(dfs).to_csv(self.path, sep="\t", index=False)


@click.command()
@verbose_option
def _main():
    from pykeen.datasets import get_dataset

    d = get_dataset(dataset=CKG)
    d.summarize()


if __name__ == "__main__":
    _main()
