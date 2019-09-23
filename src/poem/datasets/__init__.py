# -*- coding: utf-8 -*-

"""Sample datasets for use with POEM, borrowed from https://github.com/ZhenfengLei/KGDatasets.

====  ========  ===============================
  ..  Name      Reference
====  ========  ===============================
   1  fb15k     :class:`poem.datasets.fb15k`
   2  fb15k237  :class:`poem.datasets.fb15k237`
   3  kinship   :class:`poem.datasets.kinship`
   4  nations   :class:`poem.datasets.nations`
   5  umls      :class:`poem.datasets.umls`
   6  wn18      :class:`poem.datasets.wn18`
   7  wn18rr    :class:`poem.datasets.wn18rr`
   8  yago310   :class:`poem.datasets.yago310`
====  ========  ===============================

.. note:: This table can be re-generated with ``poem ls datasets -f rst``
"""

from typing import Mapping, Optional, Tuple, Union

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
    'data_sets',
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
    'get_data_set',
]

#: A mapping of data sets' names to their instances
data_sets: Mapping[str, DataSet] = dict(
    nations=nations,
    kinship=kinship,
    umls=umls,
    fb15k=fb15k,
    fb15k237=fb15k237,
    wn18=wn18,
    wn18rr=wn18rr,
    yago310=yago310,
)


def get_data_set(
    data_set: Union[None, str, DataSet] = None,
    training_triples_factory: Optional[TriplesFactory] = None,
    testing_triples_factory: Optional[TriplesFactory] = None,
    validation_triples_factory: Optional[TriplesFactory] = None,
) -> Tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
    """Get the data set."""
    if data_set is not None:
        if any(f is not None for f in (training_triples_factory, testing_triples_factory, validation_triples_factory)):
            raise ValueError('Can not specify both dataset and any triples factory.')

        if isinstance(data_set, str):
            try:
                data_set = data_sets[normalize_string(data_set)]
            except KeyError:
                raise ValueError(f'Invalid dataset name: {data_set}')

        elif not isinstance(data_set, DataSet):
            raise TypeError(f'Data set is wrong type: {type(data_set)}')

        return (
            data_set.training,
            data_set.testing,
            data_set.validation,
        )

    elif testing_triples_factory is None or training_triples_factory is None:
        raise ValueError('Must specify either dataset or both training_triples_factory and testing_triples_factory.')

    return (
        training_triples_factory,
        testing_triples_factory,
        validation_triples_factory,
    )
