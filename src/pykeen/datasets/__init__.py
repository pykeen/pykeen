"""Built-in datasets for PyKEEN.

New datasets (inheriting from :class:`~pykeen.datasets.Dataset`) can be registered with PyKEEN using the
:mod:`pykeen.datasets` group in Python entrypoints in your own `setup.py`, `setup.cfg`, `pyproject.toml`, or other
package configuration. They are loaded automatically with :func:`importlib.metadata.entry_points` via
:mod:`class_resolver`.
"""

import logging

from class_resolver import ClassResolver

from .aristo import AristoV4
from .base import (
    CompressedSingleDataset,
    Dataset,
    EagerDataset,
    LazyDataset,
    PackedRemoteDataSet,
    PackedZipRemoteDataset,
    PathDataset,
    SingleTabbedDataset,
    SourceDataSet,
    TabbedDataset,
    TarFileRemoteDataset,
    TarFileSingleDataset,
    UnpackedRemoteDataset,
    ZipSingleDataset,
)
from .biokg import BioKG
from .ckg import CKG
from .codex import CoDExLarge, CoDExMedium, CoDExSmall
from .conceptnet import ConceptNet
from .countries import Countries
from .cskg import CSKG
from .db100k import DB100K
from .dbpedia import DBpedia50
from .drkg import DRKG
from .ea import CN3l, EADataset, MTransEDataset, OpenEA, WK3l15k, WK3l120k
from .freebase import FB15k, FB15k237
from .globi import Globi
from .hetionet import Hetionet
from .kinships import Kinships
from .literal_base import NumericPathDataset
from .nations import Nations
from .ogb import OGBBioKG, OGBLoader, OGBWikiKG2
from .openbiolink import OpenBioLink, OpenBioLinkLQ
from .pharmebinet import PharMeBINet
from .pharmkg import PharmKG, PharmKG8k
from .primekg import PrimeKG
from .umls import UMLS
from .utils import get_dataset
from .wd50k import WD50KT
from .wikidata5m import Wikidata5M
from .wordnet import WN18, WN18RR
from .yago import YAGO310

__all__ = [
    "CKG",
    "CSKG",
    "DB100K",
    "DRKG",
    "UMLS",
    "WD50KT",
    "WN18",
    "WN18RR",
    "YAGO310",
    "AristoV4",
    "BioKG",
    "CN3l",
    "CoDExLarge",
    "CoDExMedium",
    "CoDExSmall",
    "ConceptNet",
    "Countries",
    "DBpedia50",
    "Dataset",
    "FB15k",
    "FB15k237",
    "Globi",
    "Hetionet",
    "Kinships",
    "Nations",
    "OGBBioKG",
    "OGBWikiKG2",
    "OpenBioLink",
    "OpenBioLinkLQ",
    "OpenEA",
    "PharMeBINet",
    "PharmKG",
    "PharmKG8k",
    "PrimeKG",
    "WK3l15k",
    "WK3l120k",
    "Wikidata5M",
    "dataset_resolver",
    "get_dataset",
    "has_dataset",
]

logger = logging.getLogger(__name__)

#: A resolver for datasets
dataset_resolver: ClassResolver[Dataset] = ClassResolver.from_subclasses(
    base=Dataset,
    skip={
        EagerDataset,
        SourceDataSet,
        LazyDataset,
        PathDataset,
        PackedRemoteDataSet,
        UnpackedRemoteDataset,
        TarFileRemoteDataset,
        PackedZipRemoteDataset,
        CompressedSingleDataset,
        TarFileSingleDataset,
        ZipSingleDataset,
        TabbedDataset,
        SingleTabbedDataset,
        NumericPathDataset,
        MTransEDataset,
        OGBLoader,
        EADataset,
    },
)
dataset_resolver.register_entrypoint("pykeen.datasets")


def has_dataset(key: str) -> bool:
    """Return if the dataset is registered in PyKEEN."""
    return dataset_resolver.lookup(key) is not None
