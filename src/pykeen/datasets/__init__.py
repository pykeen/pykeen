# -*- coding: utf-8 -*-

"""Built-in datasets for PyKEEN.

New datasets (inheriting from :class:`pykeen.datasets.Dataset`) can be registered with PyKEEN using the
:mod:`pykeen.datasets` group in Python entrypoints in your own `setup.py` or `setup.cfg` package configuration.
They are loaded automatically with :func:`pkg_resources.iter_entry_points`.
"""

import logging
from textwrap import dedent

from class_resolver import ClassResolver

from .base import (  # noqa:F401
    Dataset,
    EagerDataset,
    LazyDataset,
    PackedZipRemoteDataset,
    PathDataset,
    RemoteDataset,
    SingleTabbedDataset,
    TarFileRemoteDataset,
    UnpackedRemoteDataset,
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
from .freebase import FB15k, FB15k237
from .hetionet import Hetionet
from .kinships import Kinships
from .nations import Nations
from .ogb import OGBBioKG, OGBWikiKG2
from .openbiolink import OpenBioLink, OpenBioLinkLQ
from .openea import OpenEA
from .pharmkg import PharmKG, PharmKG8k
from .umls import UMLS
from .utils import get_dataset
from .wd50k import WD50KT
from .wikidata5m import Wikidata5M
from .wk3l import WK3l15k
from .wordnet import WN18, WN18RR
from .yago import YAGO310

__all__ = [
    # Utilities
    "dataset_resolver",
    "get_dataset",
    "has_dataset",
    # Base Classes
    "Dataset",
    # Concrete Classes
    "Hetionet",
    "Kinships",
    "Nations",
    "OpenBioLink",
    "OpenBioLinkLQ",
    "CoDExSmall",
    "CoDExMedium",
    "CoDExLarge",
    "OGBBioKG",
    "OGBWikiKG2",
    "UMLS",
    "FB15k",
    "FB15k237",
    "WK3l15k",
    "WN18",
    "WN18RR",
    "YAGO310",
    "DRKG",
    "BioKG",
    "ConceptNet",
    "CKG",
    "CSKG",
    "DBpedia50",
    "DB100K",
    "OpenEA",
    "Countries",
    "WD50KT",
    "Wikidata5M",
    "PharmKG8k",
    "PharmKG",
]

logger = logging.getLogger(__name__)

dataset_resolver = ClassResolver.from_entrypoint(group="pykeen.datasets", base=Dataset)
if not dataset_resolver.lookup_dict:
    raise RuntimeError(
        dedent(
            """\
    Datasets have been loaded with entrypoints since PyKEEN v1.0.5, which is now a
    very old version of PyKEEN.

    If you simply use `python3 -m pip install --upgrade pykeen`, the entrypoints will
    not be reloaded. Instead, please reinstall PyKEEN using the following commands:

    $ python3 -m pip uninstall pykeen
    $ python3 -m pip install pykeen

    If you are on Kaggle or Google Colab, please follow these instructions:
    https://pykeen.readthedocs.io/en/stable/installation.html#google-colab-and-kaggle-users

    If issues with Kaggle or Colab persist, please join the conversation at
    https://github.com/pykeen/pykeen/issues/373
    """
        )
    )


def has_dataset(key: str) -> bool:
    """Return if the dataset is registered in PyKEEN."""
    return dataset_resolver.lookup(key) is not None
