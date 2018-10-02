# -*- coding: utf-8 -*-

"""Script for initializing the knowledge graph embedding models."""

from keen.constants import KG_EMBEDDING_MODEL_NAME, TRANS_E_NAME, TRANS_H_NAME, TRANS_D_NAME, TRANS_R_NAME, CONV_E_NAME, \
    SE_NAME, UM_NAME, DISTMULT_NAME, ERMLP_NAME, \
    RESCAL_NAME
from keen.kg_embeddings_model.conv_e import ConvE
from keen.kg_embeddings_model.distmult import DistMult
from keen.kg_embeddings_model.ermlp import ERMLP
from keen.kg_embeddings_model.rescal import RESCAL
from keen.kg_embeddings_model.structured_embedding import StructuredEmbedding

from keen.kg_embeddings_model.trans_d import TransD
from keen.kg_embeddings_model.trans_e import TransE
from keen.kg_embeddings_model.trans_h import TransH
from keen.kg_embeddings_model.trans_r import TransR
from keen.kg_embeddings_model.unstructured_model import UnstructuredModel


def get_kg_embedding_model(config):
    """

    :param config:
    :return:
    """
    model_name = config[KG_EMBEDDING_MODEL_NAME]

    if model_name == TRANS_E_NAME:
        return TransE(config=config)

    if model_name == TRANS_H_NAME:
        return TransH(config=config)

    if model_name == TRANS_D_NAME:
        return TransD(config=config)

    if model_name == TRANS_R_NAME:
        return TransR(config=config)

    if model_name == SE_NAME:
        return StructuredEmbedding(config=config)

    if model_name == UM_NAME:
        return UnstructuredModel(config=config)

    if model_name == DISTMULT_NAME:
        return DistMult(config=config)

    if model_name == ERMLP_NAME:
        return ERMLP(config=config)
    if model_name == RESCAL_NAME:
        return RESCAL(config=config)

    if model_name == CONV_E_NAME:
        return ConvE(config=config)
