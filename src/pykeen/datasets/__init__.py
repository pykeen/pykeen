# -*- coding: utf-8 -*-

"""Sample datasets for use with PyKEEN, borrowed from https://github.com/ZhenfengLei/KGDatasets.

New datasets (inheriting from :class:`pykeen.datasets.base.Dataset`) can be registered with PyKEEN using the
:mod:`pykeen.datasets` group in Python entrypoints in your own `setup.py` or `setup.cfg` package configuration.
They are loaded automatically with :func:`pkg_resources.iter_entry_points`.
"""

import logging
import os
from typing import Any, Mapping, Optional, Set, Type, Union

from pkg_resources import iter_entry_points

from .base import (  # noqa:F401
    Dataset, EagerDataset, LazyDataset, PackedZipRemoteDataset, PathDataset, RemoteDataset, SingleTabbedDataset,
    TarFileRemoteDataset, UnpackedRemoteDataset, ZipFileRemoteDataset,
)
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
from .openbiolink import OpenBioLink, OpenBioLinkF1, OpenBioLinkF2, OpenBioLinkLQ
from .umls import UMLS
from .wordnet import WN18, WN18RR
from .yago import YAGO310
from ..triples import CoreTriplesFactory, TriplesFactory
from ..utils import normalize_string

__all__ = [
    'Hetionet',
    'Kinships',
    'Nations',
    'OpenBioLink',
    'OpenBioLinkF1',
    'OpenBioLinkF2',
    'OpenBioLinkLQ',
    'CoDExSmall',
    'CoDExMedium',
    'CoDExLarge',
    'OGBBioKG',
    'OGBWikiKG',
    'UMLS',
    'FB15k',
    'FB15k237',
    'WN18',
    'WN18RR',
    'YAGO310',
    'DRKG',
    'ConceptNet',
    'CKG',
    'CSKG',
    'DBpedia50',
    'DB100K',
    'Countries',
    'get_dataset',
    'has_dataset',
]

logger = logging.getLogger(__name__)

_DATASETS: Set[Type[Dataset]] = {
    entry.load()
    for entry in iter_entry_points(group='pykeen.datasets')
}
if not _DATASETS:
    raise RuntimeError('Datasets have been loaded with entrypoints since PyKEEN v1.0.5. Please reinstall.')

#: A mapping of datasets' names to their classes
datasets: Mapping[str, Type[Dataset]] = {
    normalize_string(cls.__name__): cls
    for cls in _DATASETS
}


def get_dataset(
    *,
    dataset: Union[None, str, Dataset, Type[Dataset]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training: Union[None, str, TriplesFactory] = None,
    testing: Union[None, str, TriplesFactory] = None,
    validation: Union[None, str, TriplesFactory] = None,
) -> Dataset:
    """Get the dataset.

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
        raise ValueError('Must specify either dataset or both training/testing triples factories')

    if dataset is not None and (training is not None or testing is not None):
        raise ValueError('Can not specify both dataset and training/testing triples factories.')

    if isinstance(dataset, Dataset):
        if dataset_kwargs:
            logger.warning('dataset_kwargs not used since a pre-instantiated dataset was given')
        return dataset

    if isinstance(dataset, str):
        if has_dataset(dataset):
            dataset: Type[Dataset] = datasets[normalize_string(dataset)]  # type: ignore
        elif not os.path.exists(dataset):
            raise ValueError(f'dataset is neither a pre-defined dataset string nor a filepath: {dataset}')
        else:
            return Dataset.from_path(dataset)

    if isinstance(dataset, type) and issubclass(dataset, Dataset):
        return dataset(**(dataset_kwargs or {}))  # type: ignore

    if dataset is not None:
        raise TypeError(f'Dataset is invalid type: {type(dataset)}')

    if isinstance(training, str) and isinstance(testing, str):
        if validation is None or isinstance(validation, str):
            return PathDataset(
                training_path=training,
                testing_path=testing,
                validation_path=validation,
                **(dataset_kwargs or {}),
            )
        elif validation is not None:
            raise TypeError(f'Validation is invalid type: {type(validation)}')

    if isinstance(training, CoreTriplesFactory) and isinstance(testing, CoreTriplesFactory):
        if validation is not None and not isinstance(validation, CoreTriplesFactory):
            raise TypeError(f'Validation is invalid type: {type(validation)}')
        if dataset_kwargs:
            logger.warning('dataset_kwargs are disregarded when passing pre-instantiated triples factories')
        return EagerDataset(
            training=training,
            testing=testing,
            validation=validation,
        )

    raise TypeError(
        f'''Training and testing must both be given as strings or Triples Factories.
        - Training: {type(training)}: {training}
        - Testing: {type(testing)}: {testing}
        ''',
    )


def has_dataset(key: str) -> bool:
    """Return if the dataset is registered in PyKEEN."""
    return normalize_string(key) in datasets
