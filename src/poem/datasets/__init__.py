# -*- coding: utf-8 -*-

"""Sample datasets for use with POEM, borrowed from https://github.com/ZhenfengLei/KGDatasets.

+---------------+------------------------------------+
| Data Set Name | Reference                          |
+===============+====================================+
| Nations       | :py:class:`poem.datasets.Nations`  |
+---------------+------------------------------------+
| Kinship       | :py:class:`poem.datasets.Kinship`  |
+---------------+------------------------------------+
| UMLS          | :py:class:`poem.datasets.Umls`     |
+---------------+------------------------------------+
| FB15K         | :py:class:`poem.datasets.FB15k`    |
+---------------+------------------------------------+
| FB15K237      | :py:class:`poem.datasets.FB15k237` |
+---------------+------------------------------------+
| WN18          | :py:class:`poem.datasets.WN18`     |
+---------------+------------------------------------+
| WN18R         | :py:class:`poem.datasets.WN18RR`   |
+---------------+------------------------------------+
| YAGO3-10      | :py:class:`poem.datasets.YAGO310`  |
+---------------+------------------------------------+
"""

from typing import Mapping

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
from .yago import YAGO310, yago3_10

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
