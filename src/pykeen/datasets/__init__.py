# -*- coding: utf-8 -*-

"""Sample datasets for use with PyKEEN, borrowed from https://github.com/ZhenfengLei/KGDatasets.

New datasets (inheriting from :class:`DataSet`) can be registered with PyKEEN using the
`pykeen.datasets` group in Python entrypoints in your own `setup.py` or `setup.cfg` package configuration.
They are loaded automatically with :func:`pkg_resources.iter_entry_points`.
"""

from typing import Any, Mapping, Optional, Set, Tuple, Type, Union

from pkg_resources import iter_entry_points

from .base import (  # noqa:F401
    DataSet, LazyDataSet, PackedZipRemoteDataSet, PathDataSet, RemoteDataSet, SingleTabbedDataset, TarFileRemoteDataSet,
    ZipFileRemoteDataSet,
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

_DATASETS: Set[Type[DataSet]] = {
    entry.load()
    for entry in iter_entry_points(group='pykeen.datasets')
}

#: A mapping of datasets' names to their classes
datasets: Mapping[str, Type[DataSet]] = normalized_lookup(_DATASETS)


def get_dataset(
    *,
    dataset: Union[None, str, Type[DataSet]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training_triples_factory: Optional[TriplesFactory] = None,
    testing_triples_factory: Optional[TriplesFactory] = None,
    validation_triples_factory: Optional[TriplesFactory] = None,
) -> Tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
    """Get the dataset."""
    if dataset is not None:
        if any(f is not None for f in (training_triples_factory, testing_triples_factory, validation_triples_factory)):
            raise ValueError('Can not specify both dataset and any triples factory.')

        if isinstance(dataset, str):
            if normalize_string(dataset) in datasets:
                dataset: Type[DataSet] = datasets[normalize_string(dataset)]
            else:  # assume its a file path
                _tf = TriplesFactory(path=dataset)
                train, test, valid = _tf.split([0.8, 0.1, 0.1])
                return train, test, valid

        elif not isinstance(dataset, type) or not issubclass(dataset, DataSet):
            raise TypeError(f'Data set is wrong type: {type(dataset)}')

        dataset_instance = dataset(
            **(dataset_kwargs or {}),
        )
        return dataset_instance.factories

    elif testing_triples_factory is None or training_triples_factory is None:
        raise ValueError('Must specify either dataset or both training_triples_factory and testing_triples_factory.')

    return (
        training_triples_factory,
        testing_triples_factory,
        validation_triples_factory,
    )
