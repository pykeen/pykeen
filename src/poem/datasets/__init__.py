# -*- coding: utf-8 -*-

"""Sample datasets for use with POEM, borrowed from https://github.com/ZhenfengLei/KGDatasets."""

from .dataset import DataSet
from .kinship import (
    KinshipTestingTriplesFactory, KinshipTrainingTriplesFactory, KinshipValidationTriplesFactory,
    kinship,
)
from .nations import (
    NationsTestingTriplesFactory, NationsTrainingTriplesFactory, NationsValidationTriplesFactory,
    nations,
)
from .umls import UmlsTestingTriplesFactory, UmlsTrainingTriplesFactory, UmlsValidationTriplesFactory, umls

__all__ = [
    'DataSet',
    'datasets',
    'kinship',
    'KinshipTrainingTriplesFactory',
    'KinshipTestingTriplesFactory',
    'KinshipValidationTriplesFactory',
    'nations',
    'NationsTrainingTriplesFactory',
    'NationsValidationTriplesFactory',
    'NationsTestingTriplesFactory',
    'umls',
    'UmlsTrainingTriplesFactory',
    'UmlsTestingTriplesFactory',
    'UmlsValidationTriplesFactory',
]

#: A maintained dictionary of pre-packaged data sets
datasets = dict(
    nations=nations,
    kinship=kinship,
    umls=umls,
)
