# -*- coding: utf-8 -*-

"""Sample datasets for use with POEM, borrowed from https://github.com/ZhenfengLei/KGDatasets."""

from .dataset import DataSet
from .freebase import fb15k, fb15k237
from .kinship import (
    KinshipTestingTriplesFactory, KinshipTrainingTriplesFactory, KinshipValidationTriplesFactory,
    kinship,
)
from .nations import (
    NationsTestingTriplesFactory, NationsTrainingTriplesFactory, NationsValidationTriplesFactory,
    nations,
)
from .umls import UmlsTestingTriplesFactory, UmlsTrainingTriplesFactory, UmlsValidationTriplesFactory, umls
from .wordnet import wn18, wn18rr
from .yago import yago3_10

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
    'fb15k',
    'fb15k237',
    'wn18',
    'wn18rr',
    'yago3_10',
]

#: A maintained dictionary of pre-packaged data sets
datasets = dict(
    nations=nations,
    kinship=kinship,
    umls=umls,
    fb15k=fb15k,
    fb15k237=fb15k237,
    wn18=wn18,
    wn18rr=wn18rr,
    yago3_10=yago3_10,
)
