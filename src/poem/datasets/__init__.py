# -*- coding: utf-8 -*-

"""Sample datasets for use with POEM, borrowed from https://github.com/ZhenfengLei/KGDatasets."""

from .kinship import KinshipTestingTriplesFactory, KinshipTrainingTriplesFactory, KinshipValidationTriplesFactory
from .nations import NationsTestingTriplesFactory, NationsTrainingTriplesFactory, NationsValidationTriplesFactory
from .umls import UmlsTestingTriplesFactory, UmlsTrainingTriplesFactory, UmlsValidationTriplesFactory

__all__ = [
    'KinshipTrainingTriplesFactory',
    'KinshipTestingTriplesFactory',
    'KinshipValidationTriplesFactory',
    'NationsTrainingTriplesFactory',
    'NationsValidationTriplesFactory',
    'NationsTestingTriplesFactory',
    'UmlsTrainingTriplesFactory',
    'UmlsTestingTriplesFactory',
    'UmlsValidationTriplesFactory',
]
