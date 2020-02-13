# -*- coding: utf-8 -*-

"""Sample datasets for use with PyKEEN, borrowed from https://github.com/ZhenfengLei/KGDatasets.

========  =================================
Name      Reference
========  =================================
fb15k     :class:`pykeen.datasets.fb15k`
fb15k237  :class:`pykeen.datasets.fb15k237`
kinship   :class:`pykeen.datasets.kinship`
nations   :class:`pykeen.datasets.nations`
umls      :class:`pykeen.datasets.umls`
wn18      :class:`pykeen.datasets.wn18`
wn18rr    :class:`pykeen.datasets.wn18rr`
yago310   :class:`pykeen.datasets.yago310`
========  =================================

.. note:: This table can be re-generated with ``pykeen ls datasets -f rst``
"""

from typing import Any, Mapping, Optional, Set, Tuple, Type, Union

from .dataset import DataSet
from .freebase import FB15k, FB15k237, fb15k, fb15k237
from .kinship import (
    Kinship, KinshipTestingTriplesFactory, KinshipTrainingTriplesFactory, KinshipValidationTriplesFactory, kinship,
)
from .nations import (
    Nations, NationsTestingTriplesFactory, NationsTrainingTriplesFactory, NationsValidationTriplesFactory, nations,
)
from .umls import Umls, UmlsTestingTriplesFactory, UmlsTrainingTriplesFactory, UmlsValidationTriplesFactory, umls
from .wordnet import WN18, WN18RR, wn18, wn18rr
from .yago import YAGO310, yago310
from ..triples import TriplesFactory
from ..utils import normalize_string

__all__ = [
    'DataSet',
    'datasets',
    'kinship',
    'Kinship',
    'KinshipTrainingTriplesFactory',
    'KinshipTestingTriplesFactory',
    'KinshipValidationTriplesFactory',
    'nations',
    'Nations',
    'NationsTrainingTriplesFactory',
    'NationsValidationTriplesFactory',
    'NationsTestingTriplesFactory',
    'umls',
    'Umls',
    'UmlsTrainingTriplesFactory',
    'UmlsTestingTriplesFactory',
    'UmlsValidationTriplesFactory',
    'FB15k',
    'fb15k',
    'FB15k237',
    'fb15k237',
    'WN18',
    'wn18',
    'WN18RR',
    'wn18rr',
    'YAGO310',
    'yago310',
    'get_dataset',
]

_DATASETS: Set[Type[DataSet]] = {
    Nations,
    Kinship,
    Umls,
    FB15k,
    FB15k237,
    WN18,
    WN18RR,
    YAGO310,
}

#: A mapping of data sets' names to their classes
datasets: Mapping[str, Type[DataSet]] = {
    normalize_string(cls.__name__): cls
    for cls in _DATASETS
}


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
            try:
                dataset: Type[DataSet] = datasets[normalize_string(dataset)]
            except KeyError:
                raise ValueError(f'Invalid dataset name: {dataset}')

        elif not isinstance(dataset, type) or not issubclass(dataset, DataSet):
            raise TypeError(f'Data set is wrong type: {type(dataset)}')

        dataset_instance = dataset(
            **(dataset_kwargs or {})
        )
        return dataset_instance.factories

    elif testing_triples_factory is None or training_triples_factory is None:
        raise ValueError('Must specify either dataset or both training_triples_factory and testing_triples_factory.')

    return (
        training_triples_factory,
        testing_triples_factory,
        validation_triples_factory,
    )
