# -*- coding: utf-8 -*-

"""Sample datasets for use with PyKEEN, borrowed from https://github.com/ZhenfengLei/KGDatasets.

New datasets (inheriting from :class:`pykeen.datasets.base.Dataset`) can be registered with PyKEEN using the
:mod:`pykeen.datasets` group in Python entrypoints in your own `setup.py` or `setup.cfg` package configuration.
They are loaded automatically with :func:`pkg_resources.iter_entry_points`.
"""

import logging
import pathlib
from typing import Any, Mapping, Optional, Type, Union

from class_resolver import Resolver

from .base import (  # noqa:F401
    Dataset, EagerDataset, LazyDataset, PackedZipRemoteDataset, PathDataset, RemoteDataset, SingleTabbedDataset,
    TarFileRemoteDataset, UnpackedRemoteDataset,
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
from .openbiolink import OpenBioLink, OpenBioLinkLQ
from .umls import UMLS
from .wk3l import WK3l15k
from .wordnet import WN18, WN18RR
from .yago import YAGO310
from ..triples import CoreTriplesFactory

__all__ = [
    # Concrete Classes
    'Hetionet',
    'Kinships',
    'Nations',
    'OpenBioLink',
    'OpenBioLinkLQ',
    'CoDExSmall',
    'CoDExMedium',
    'CoDExLarge',
    'OGBBioKG',
    'OGBWikiKG',
    'UMLS',
    'FB15k',
    'FB15k237',
    'WK3l15k',
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
    # Utilities
    'dataset_resolver',
    'get_dataset',
    'has_dataset',
]

logger = logging.getLogger(__name__)

dataset_resolver = Resolver.from_entrypoint(group='pykeen.datasets', base=Dataset)
if not dataset_resolver.lookup_dict:
    raise RuntimeError('Datasets have been loaded with entrypoints since PyKEEN v1.0.5. Please reinstall.')


def get_dataset(
    *,
    dataset: Union[None, str, pathlib.Path, Dataset, Type[Dataset]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training: Union[None, str, pathlib.Path, CoreTriplesFactory] = None,
    testing: Union[None, str, pathlib.Path, CoreTriplesFactory] = None,
    validation: Union[None, str, pathlib.Path, CoreTriplesFactory] = None,
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

    if isinstance(dataset, pathlib.Path):
        return Dataset.from_path(dataset)

    if isinstance(dataset, str):
        if has_dataset(dataset):
            return dataset_resolver.make(dataset, dataset_kwargs)
        else:
            # Assume it's a file path
            return Dataset.from_path(dataset)

    if isinstance(dataset, type) and issubclass(dataset, Dataset):
        return dataset(**(dataset_kwargs or {}))  # type: ignore

    if dataset is not None:
        raise TypeError(f'Dataset is invalid type: {type(dataset)}')

    if isinstance(training, (str, pathlib.Path)) and isinstance(testing, (str, pathlib.Path)):
        if validation is None or isinstance(validation, (str, pathlib.Path)):
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
    return dataset_resolver.lookup(key) is not None
