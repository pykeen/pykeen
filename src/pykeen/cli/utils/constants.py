# -*- coding: utf-8 -*-

"""Constants defined for PyKEEN's CLI."""

from collections import OrderedDict

from pykeen.constants import ADAGRAD_OPTIMIZER_NAME, ADAM_OPTIMIZER_NAME, SGD_OPTIMIZER_NAME
from ...kge_models import (
    ConvE, DistMult, ERMLP, RESCAL, StructuredEmbedding, TransD, TransE, TransH, TransR, UnstructuredModel,
)

__all__ = [
    'ID_TO_KG_MODEL_MAPPING',
    'KG_MODEL_TO_ID_MAPPING',
    'ID_TO_OPTIMIZER_MAPPING',
    'OPTIMIZER_TO_ID_MAPPING',
]

ID_TO_KG_MODEL_MAPPING = OrderedDict({
    '1': TransE.model_name,
    '2': TransH.model_name,
    '3': TransR.model_name,
    '4': TransD.model_name,
    '5': StructuredEmbedding.model_name,
    '6': UnstructuredModel.model_name,
    '7': DistMult.model_name,
    '8': ERMLP.model_name,
    '9': RESCAL.model_name,
    '10': ConvE.model_name,
})

KG_MODEL_TO_ID_MAPPING = OrderedDict({
    value: key
    for key, value in ID_TO_KG_MODEL_MAPPING.items()
})

ID_TO_OPTIMIZER_MAPPING = OrderedDict({
    '1': SGD_OPTIMIZER_NAME,
    '2': ADAGRAD_OPTIMIZER_NAME,
    '3': ADAM_OPTIMIZER_NAME,
})

OPTIMIZER_TO_ID_MAPPING = OrderedDict({
    value: key
    for key, value in ID_TO_OPTIMIZER_MAPPING.items()
})
