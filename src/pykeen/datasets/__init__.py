# -*- coding: utf-8 -*-

"""Sample datasets for use with PyKEEN, borrowed from https://github.com/ZhenfengLei/KGDatasets."""

import logging
import os
from typing import Any, Mapping, Optional, Set, Type, Union

from .base import (  # noqa:F401
    DataSet, EagerDataset, LazyDataSet, PackedZipRemoteDataSet, PathDataSet, RemoteDataSet, SingleTabbedDataset,
    TarFileRemoteDataSet, ZipFileRemoteDataSet,
)
from .freebase import FB15k, FB15k237
from .hetionet import Hetionet
from .kinships import Kinships
from .nations import Nations
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
    'UMLS',
    'FB15k',
    'FB15k237',
    'WN18',
    'WN18RR',
    'YAGO310',
    'get_dataset',
]

logger = logging.getLogger(__name__)

_DATASETS: Set[Type[DataSet]] = {
    Nations,
    Kinships,
    UMLS,
    FB15k,
    FB15k237,
    Hetionet,
    OpenBioLink,
    OpenBioLinkF1,
    OpenBioLinkF2,
    OpenBioLinkLQ,
    WN18,
    WN18RR,
    YAGO310,
}

#: A mapping of datasets' names to their classes
datasets: Mapping[str, Type[DataSet]] = normalized_lookup(_DATASETS)


def get_dataset(
    *,
    dataset: Union[None, str, DataSet, Type[DataSet]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training_triples_factory: Union[None, str, TriplesFactory] = None,
    testing_triples_factory: Union[None, str, TriplesFactory] = None,
    validation_triples_factory: Union[None, str, TriplesFactory] = None,
) -> DataSet:
    """Get the dataset.

    :raises ValueError:
    :raises TypeError:
    """
    if dataset is None and (training_triples_factory is None or testing_triples_factory is None):
        raise ValueError('Must specify either dataset or both training/testing triples factories')

    if dataset is not None and (training_triples_factory is not None or testing_triples_factory is not None):
        raise ValueError('Can not specify both dataset and training/testing triples factories.')

    if isinstance(dataset, DataSet):
        if dataset_kwargs:
            logger.warning('dataset_kwargs not used since a pre-instantiated dataset was given')
        return dataset

    if isinstance(dataset, str):
        normalized_dataset = normalize_string(dataset)
        if normalized_dataset in datasets:
            dataset: Type[DataSet] = datasets[normalized_dataset]
        elif not os.path.exists(dataset):
            raise ValueError('dataset is neither a pre-defined dataset string nor a filepath')
        else:
            return DataSet.from_path(dataset)

    if isinstance(dataset, type) and issubclass(dataset, DataSet):
        return dataset(**(dataset_kwargs or {}))

    if dataset is not None:
        raise TypeError(f'Data set is invalid type: {type(dataset)}')

    if isinstance(training_triples_factory, str) and isinstance(testing_triples_factory, str):
        if validation_triples_factory is not None and not isinstance(validation_triples_factory, str):
            raise TypeError(f'Validation is invalid type: {type(validation_triples_factory)}')
        return PathDataSet(
            training_path=training_triples_factory,
            testing_path=testing_triples_factory,
            validation_path=validation_triples_factory,
            **(dataset_kwargs or {}),
        )

    if isinstance(training_triples_factory, TriplesFactory) and isinstance(testing_triples_factory, TriplesFactory):
        if validation_triples_factory is not None and not isinstance(validation_triples_factory, TriplesFactory):
            raise TypeError(f'Validation is invalid type: {type(validation_triples_factory)}')
        if dataset_kwargs:
            logger.warning('dataset_kwargs are disregarded when passing pre-instantiated triples factories')
        return EagerDataset(
            training=training_triples_factory,
            testing=testing_triples_factory,
            validation=validation_triples_factory,
        )

    raise TypeError('Training and testing must both be given as strings or Triples Factories')
