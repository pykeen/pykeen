# -*- coding: utf-8 -*-

"""Built-in datasets for PyKEEN.

New datasets (inheriting from :class:`pykeen.datasets.base.Dataset`) can be registered with PyKEEN using the
:mod:`pykeen.datasets` group in Python entrypoints in your own `setup.py` or `setup.cfg` package configuration.
They are loaded automatically with :func:`pkg_resources.iter_entry_points`.
"""

import base64
import hashlib
import logging
import pathlib
from textwrap import dedent
from typing import Any, Mapping, Optional, Type, Union

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
from .ogb import OGBBioKG, OGBWikiKG
from .openbiolink import OpenBioLink, OpenBioLinkLQ
from .openea import OpenEA
from .pharmkg8k import PharmKG8k
from .umls import UMLS
from .wd50k import WD50KT
from .wikidata5m import Wikidata5M
from .wk3l import WK3l15k
from .wordnet import WN18, WN18RR
from .yago import YAGO310
from ..constants import PYKEEN_DATASETS
from ..triples import CoreTriplesFactory

__all__ = [
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
    "OGBWikiKG",
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
    # Utilities
    "dataset_resolver",
    "get_dataset",
    "has_dataset",
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


def get_dataset(
    *,
    dataset: Union[None, str, pathlib.Path, Dataset, Type[Dataset]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training: Union[None, str, pathlib.Path, CoreTriplesFactory] = None,
    testing: Union[None, str, pathlib.Path, CoreTriplesFactory] = None,
    validation: Union[None, str, pathlib.Path, CoreTriplesFactory] = None,
) -> Dataset:
    """Get a dataset.

    :param dataset: The name of a dataset, an instance of a dataset, or the class for a dataset.
    :param dataset_kwargs: The keyword arguments, only to be used when a class for a dataset is used for
        the ``dataset`` keyword argument.
    :param training: A triples factory for training triples or a path to a training triples file if ``dataset=None``
    :param testing: A triples factory for testing triples or a path to a testing triples file  if ``dataset=None``
    :param validation: A triples factory for validation triples or a path to a validation triples file
        if ``dataset=None``
    :returns: An instantiated dataset

    :raises ValueError: for incorrect usage of the input of the function
    :raises TypeError: If a type is given for ``dataset`` but it's not a subclass of
        :class:`pykeen.datasets.base.Dataset`
    """
    if dataset is None and (training is None or testing is None):
        raise ValueError("Must specify either dataset or both training/testing triples factories")

    if dataset is not None and (training is not None or testing is not None):
        raise ValueError("Can not specify both dataset and training/testing triples factories.")

    if isinstance(dataset, Dataset):
        if dataset_kwargs:
            logger.warning("dataset_kwargs not used since a pre-instantiated dataset was given")
        return dataset

    if isinstance(dataset, pathlib.Path):
        return Dataset.from_path(dataset)

    if isinstance(dataset, str):
        if has_dataset(dataset):
            return _cached_get_dataset(dataset, dataset_kwargs)
        else:
            # Assume it's a file path
            return Dataset.from_path(dataset)

    if isinstance(dataset, type) and issubclass(dataset, Dataset):
        return dataset(**(dataset_kwargs or {}))  # type: ignore

    if dataset is not None:
        raise TypeError(f"Dataset is invalid type: {type(dataset)}")

    if isinstance(training, (str, pathlib.Path)) and isinstance(testing, (str, pathlib.Path)):
        if validation is None or isinstance(validation, (str, pathlib.Path)):
            return PathDataset(
                training_path=training,
                testing_path=testing,
                validation_path=validation,
                **(dataset_kwargs or {}),
            )
        elif validation is not None:
            raise TypeError(f"Validation is invalid type: {type(validation)}")

    if isinstance(training, CoreTriplesFactory) and isinstance(testing, CoreTriplesFactory):
        if validation is not None and not isinstance(validation, CoreTriplesFactory):
            raise TypeError(f"Validation is invalid type: {type(validation)}")
        if dataset_kwargs:
            logger.warning("dataset_kwargs are disregarded when passing pre-instantiated triples factories")
        return EagerDataset(
            training=training,
            testing=testing,
            validation=validation,
        )

    raise TypeError(
        f"""Training and testing must both be given as strings or Triples Factories.
        - Training: {type(training)}: {training}
        - Testing: {type(testing)}: {testing}
        """,
    )


def _digest_kwargs(dataset_kwargs: Mapping[str, Any]) -> str:
    digester = hashlib.sha256()
    for key in sorted(dataset_kwargs.keys()):
        digester.update(key.encode(encoding="utf8"))
        digester.update(str(dataset_kwargs[key]).encode(encoding="utf8"))
    return base64.urlsafe_b64encode(digester.digest()).decode("utf8")[:32]


def _cached_get_dataset(
    dataset: str,
    dataset_kwargs: Optional[Mapping[str, Any]],
    force: bool = False,
) -> Dataset:
    """Get dataset by name, potentially using file-based cache."""
    # hash kwargs
    dataset_kwargs = dataset_kwargs or {}
    digest = _digest_kwargs(dataset_kwargs)

    # normalize dataset name
    dataset = dataset_resolver.normalize(dataset)

    # get canonic path
    path = PYKEEN_DATASETS.joinpath(dataset, "cache", digest)

    # try to use cached dataset
    if path.is_dir() and not force:
        logger.info(f"Loading cached preprocessed dataset from {path.as_uri()}")
        return Dataset.from_directory_binary(path)

    # load dataset without cache
    dataset_instance = dataset_resolver.make(dataset, dataset_kwargs)

    # store cache
    logger.info(f"Caching preprocessed dataset to {path.as_uri()}")
    dataset_instance.to_directory_binary(path=path)

    return dataset_instance


def has_dataset(key: str) -> bool:
    """Return if the dataset is registered in PyKEEN."""
    return dataset_resolver.lookup(key) is not None
