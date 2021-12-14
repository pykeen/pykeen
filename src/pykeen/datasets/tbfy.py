"""The TheyBuyForYou dataset."""

import zipfile
from typing import Iterable

import pandas
from docdata import parse_docdata
from tqdm.auto import tqdm

from pykeen.constants import PYKEEN_MODULE
from pykeen.triples.triples_factory import TriplesFactory

from .base import LazyDataset

URL = "https://zenodo.org/record/5759351/files/TBFY_DATA_DUMP_RDF.zip?download=1"


def _read_turtle(zfile: zipfile.ZipFile, name: zipfile.ZipInfo) -> pandas.DataFrame:
    with zfile.open(name=name, mode="r") as nt_file:
        df = pandas.read_csv(nt_file, sep=" ", names=["head", "relation", "tail"], dtype=str)
    # strip trailing dot
    df["tail"] = df["tail"].apply(lambda x: x[:-1])
    # drop all triples with literals
    df = df[~df.applymap(lambda x: "^^" in x).any(axis=1)]
    return df


def _iter_triples_progress(zfile: zipfile.ZipFile) -> Iterable[pandas.DataFrame]:
    # 3128397 files
    with tqdm(unit="line", unit_scale=True) as progress:
        for info in zfile.infolist():
            if not info.filename.endswith(".nt"):
                continue
            df = _read_turtle(zfile=zfile, name=info)
            progress.update(df.shape[0])
            yield df


@parse_docdata
class TBFY(LazyDataset):
    """The TBFY dataset.

    ---
    name: TBFY
    citation:
        author: Soylu
        year: 2021
        link: http://www.semantic-web-journal.net/system/files/swj2618.pdf
    """

    def __init__(self):
        super().__init__()

    def _load(self) -> None:
        # we do not use PYKEEN_DATASETS, since that is already path, and does not have the ensure method anymore
        path = PYKEEN_MODULE.ensure(
            "datasets",
            self.__class__.__name__.lower(),
            url=URL,
            name="TBFY.zip",
            download_kwargs=dict(
                hexdigests=dict(md5="bebfee875082063e61154111f6f0e0c2"),
            ),
        )
        with zipfile.ZipFile(file=path, mode="r") as zfile:
            df = pandas.concat(
                _iter_triples_progress(zfile=zfile),
                ignore_index=True,
            )
        self._training, self._validation, self._testing = TriplesFactory.from_labeled_triples(
            triples=df.values,
        ).split(ratios=[0.8, 0.1])

    def _load_validation(self) -> None:
        pass
