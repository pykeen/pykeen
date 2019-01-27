# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from typing import Dict

from torch.nn import Module

from pykeen.constants import KG_EMBEDDING_MODEL_NAME
from pykeen.kge_models import (
    ConvE, DistMult, ERMLP, RESCAL, StructuredEmbedding, TransD, TransE, TransH, TransR, UnstructuredModel,
)

__all__ = [
    'KGE_MODELS',
    'get_kge_model',
]

#: A mapping from KGE model names to KGE model classes
KGE_MODELS = {
    TransE.model_name: TransE,
    TransH.model_name: TransH,
    TransD.model_name: TransD,
    TransR.model_name: TransR,
    StructuredEmbedding.model_name: StructuredEmbedding,
    UnstructuredModel.model_name: UnstructuredModel,
    DistMult.model_name: DistMult,
    ERMLP.model_name: ERMLP,
    RESCAL.model_name: RESCAL,
    ConvE.model_name: ConvE,
}


def get_kge_model(config: Dict) -> Module:
    """Get an instance of a knowledge graph embedding model with the given configuration."""
    kge_model_name = config[KG_EMBEDDING_MODEL_NAME]
    kge_model_cls = KGE_MODELS.get(kge_model_name)

    if kge_model_cls is None:
        raise ValueError(f'Invalid KGE model name: {kge_model_name}')

    return kge_model_cls(config=config)
