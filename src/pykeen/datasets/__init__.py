# -*- coding: utf-8 -*-

"""Sample datasets for use with PyKEEN, borrowed from https://github.com/ZhenfengLei/KGDatasets.

New datasets (inheriting from :class:`pykeen.datasets.base.DataSet`) can be registered with PyKEEN using the
`pykeen.datasets` group in Python entrypoints in your own `setup.py` or `setup.cfg` package configuration.
They are loaded automatically with :func:`pkg_resources.iter_entry_points`.
"""

import logging
import os
from typing import Any, Mapping, Optional, Set, Type, Union

from pkg_resources import iter_entry_points

from .base import (  # noqa:F401
    DataSet, EagerDataset, LazyDataSet, PackedZipRemoteDataSet, PathDataSet, RemoteDataSet, SingleTabbedDataset,
    TarFileRemoteDataSet, UnpackedRemoteDataSet, ZipFileRemoteDataSet,
)
from .codex import CoDExLarge, CoDExMedium, CoDExSmall
from .freebase import FB15k, FB15k237
from .hetionet import Hetionet
from .kinships import Kinships
from .nations import Nations
from .ogb import OGBBioKG, OGBWikiKG
from .openbiolink import OpenBioLink, OpenBioLinkF1, OpenBioLinkF2, OpenBioLinkLQ
from .umls import UMLS
from .wordnet import WN18, WN18RR
from .yago import YAGO310
from ..triples import TriplesFactory
from ..utils import normalize_string, normalized_lookup

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
    'get_dataset',
    'has_dataset',
]

logger = logging.getLogger(__name__)

_DATASETS: Set[Type[DataSet]] = {
    entry.load()
    for entry in iter_entry_points(group='pykeen.datasets')
}
if not _DATASETS:
    raise RuntimeError('Datasets have been loaded with entrypoints since PyKEEN v1.0.5. Please reinstall.')

#: A mapping of datasets' names to their classes
datasets: Mapping[str, Type[DataSet]] = normalized_lookup(_DATASETS)


def get_dataset(
    *,
    dataset: Union[None, str, DataSet, Type[DataSet]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training: Union[None, str, TriplesFactory] = None,
    testing: Union[None, str, TriplesFactory] = None,
    validation: Union[None, str, TriplesFactory] = None,
) -> DataSet:
    """Get the dataset.

    :raises ValueError:
    :raises TypeError:
    """
    if dataset is None and (training is None or testing is None):
        raise ValueError('Must specify either dataset or both training/testing triples factories')

    if dataset is not None and (training is not None or testing is not None):
        raise ValueError('Can not specify both dataset and training/testing triples factories.')

    if isinstance(dataset, DataSet):
        if dataset_kwargs:
            logger.warning('dataset_kwargs not used since a pre-instantiated dataset was given')
        return dataset

    if isinstance(dataset, str):
        if has_dataset(dataset):
            dataset: Type[DataSet] = datasets[normalize_string(dataset)]
        elif not os.path.exists(dataset):
            raise ValueError('dataset is neither a pre-defined dataset string nor a filepath')
        else:
            return DataSet.from_path(dataset)

    if isinstance(dataset, type) and issubclass(dataset, DataSet):
        return dataset(**(dataset_kwargs or {}))

    if dataset is not None:
        raise TypeError(f'Data set is invalid type: {type(dataset)}')

    if isinstance(training, str) and isinstance(testing, str):
        if validation is not None and not isinstance(validation, str):
            raise TypeError(f'Validation is invalid type: {type(validation)}')
        return PathDataSet(
            training_path=training,
            testing_path=testing,
            validation_path=validation,
            **(dataset_kwargs or {}),
        )

    if isinstance(training, TriplesFactory) and isinstance(testing, TriplesFactory):
        if validation is not None and not isinstance(validation, TriplesFactory):
            raise TypeError(f'Validation is invalid type: {type(validation)}')
        if dataset_kwargs:
            logger.warning('dataset_kwargs are disregarded when passing pre-instantiated triples factories')
        return EagerDataset(
            training=training,
            testing=testing,
            validation=validation,
        )

    raise TypeError('Training and testing must both be given as strings or Triples Factories')


def has_dataset(key: str) -> bool:
    """Return if the dataset is registered in PyKEEN."""
    return normalize_string(key) in datasets
