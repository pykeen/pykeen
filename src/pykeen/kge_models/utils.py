# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from typing import Dict

from torch.nn import Module

from pykeen.constants import (
    CONV_E_NAME, DISTMULT_NAME, ERMLP_NAME, KG_EMBEDDING_MODEL_NAME, RESCAL_NAME, SE_NAME, TRANS_D_NAME, TRANS_E_NAME,
    TRANS_H_NAME, TRANS_R_NAME, UM_NAME,
)
from pykeen.kge_models import (
    ConvE, DistMult, ERMLP, RESCAL, StructuredEmbedding, TransD, TransE, TransH, TransR, UnstructuredModel,
)

__all__ = [
    'KGE_MODELS',
    'get_kge_model',
]

KGE_MODELS = {
    TRANS_E_NAME: TransE,
    TRANS_H_NAME: TransH,
    TRANS_D_NAME: TransD,
    TRANS_R_NAME: TransR,
    SE_NAME: StructuredEmbedding,
    UM_NAME: UnstructuredModel,
    DISTMULT_NAME: DistMult,
    ERMLP_NAME: ERMLP,
    RESCAL_NAME: RESCAL,
    CONV_E_NAME: ConvE,
}


def get_kge_model(config: Dict) -> Module:
    """Get an instance of a knowledge graph embedding model with the given configuration."""
    kge_model_name = config[KG_EMBEDDING_MODEL_NAME]
    kge_model_cls = KGE_MODELS.get(kge_model_name)

    if kge_model_cls is None:
        raise ValueError(f'Invalid KGE model name: {kge_model_name}')

    return kge_model_cls(config=config)
