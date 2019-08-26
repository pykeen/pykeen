# -*- coding: utf-8 -*-

"""Sample datasets for use with POEM, borrowed from https://github.com/ZhenfengLei/KGDatasets."""

from typing import Mapping

from .dataset import DataSet
from .freebase import FB15k, FB15k237, fb15k, fb15k237
from .kinship import (
    KinshipTestingTriplesFactory, KinshipTrainingTriplesFactory, KinshipValidationTriplesFactory, kinship,
)
from .nations import (
    NationsTestingTriplesFactory, NationsTrainingTriplesFactory, NationsValidationTriplesFactory, nations,
)
from .umls import UmlsTestingTriplesFactory, UmlsTrainingTriplesFactory, UmlsValidationTriplesFactory, umls
from .wordnet import WN18, WN18RR, wn18, wn18rr
from .yago import YAGO310, yago3_10

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
    'FB15k',
    'fb15k',
    'FB15k237',
    'fb15k237',
    'WN18',
    'wn18',
    'WN18RR',
    'wn18rr',
    'YAGO310',
    'yago3_10',
]

#: A maintained dictionary of pre-packaged data sets
datasets: Mapping[str, DataSet] = dict(
    nations=nations,
    kinship=kinship,
    umls=umls,
    fb15k=fb15k,
    fb15k237=fb15k237,
    wn18=wn18,
    wn18rr=wn18rr,
    yago3_10=yago3_10,
)
