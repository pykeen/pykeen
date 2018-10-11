# -*- coding: utf-8 -*-

"""Script for initializing the knowledge graph embedding models."""

import torch.optim as optim

from pykeen.constants import KG_EMBEDDING_MODEL_NAME, TRANS_E_NAME, TRANS_H_NAME, TRANS_D_NAME, TRANS_R_NAME, \
    CONV_E_NAME, \
    SE_NAME, UM_NAME, DISTMULT_NAME, ERMLP_NAME, \
    RESCAL_NAME, OPTMIZER_NAME, SGD_OPTIMIZER_NAME, LEARNING_RATE, ADAGRAD_OPTIMIZER_NAME, ADAM_OPTIMIZER_NAME
from pykeen.kg_embeddings_model.conv_e import ConvE
from pykeen.kg_embeddings_model.distmult import DistMult
from pykeen.kg_embeddings_model.ermlp import ERMLP
from pykeen.kg_embeddings_model.rescal import RESCAL
from pykeen.kg_embeddings_model.structured_embedding import StructuredEmbedding
from pykeen.kg_embeddings_model.trans_d import TransD
from pykeen.kg_embeddings_model.trans_e import TransE
from pykeen.kg_embeddings_model.trans_h import TransH
from pykeen.kg_embeddings_model.trans_r import TransR
from pykeen.kg_embeddings_model.unstructured_model import UnstructuredModel


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


def get_optimizer(config, kg_embedding_model):
    """

    :param config:
    :return:
    """

    optimizer_name = config[OPTMIZER_NAME]

    if optimizer_name == SGD_OPTIMIZER_NAME:
        return optim.SGD(kg_embedding_model.parameters(), lr=config[LEARNING_RATE])

    if optimizer_name == ADAGRAD_OPTIMIZER_NAME:
        optim.Adagrad(kg_embedding_model.parameters(), lr=config[LEARNING_RATE])

    if optimizer_name == ADAM_OPTIMIZER_NAME:
        optim.Adam(kg_embedding_model.parameters(), lr=config[LEARNING_RATE])
