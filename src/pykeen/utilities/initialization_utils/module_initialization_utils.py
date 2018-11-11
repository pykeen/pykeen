# -*- coding: utf-8 -*-

"""Script for initializing the knowledge graph embedding models."""

import torch.optim as optim

from pykeen.constants import (
    ADAGRAD_OPTIMIZER_NAME, ADAM_OPTIMIZER_NAME, CONV_E_NAME, DISTMULT_NAME, ERMLP_NAME, KG_EMBEDDING_MODEL_NAME,
    LEARNING_RATE, OPTMIZER_NAME, RESCAL_NAME, SE_NAME, SGD_OPTIMIZER_NAME, TRANS_D_NAME, TRANS_E_NAME, TRANS_H_NAME,
    TRANS_R_NAME, UM_NAME,
)
from pykeen.kg_embeddings_model import (
    ConvE, DistMult, ERMLP, RESCAL, StructuredEmbedding, TransD, TransE, TransH,
    TransR, UnstructuredModel,
)


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

    parameters = filter(lambda p: p.requires_grad, kg_embedding_model.parameters())

    if optimizer_name == SGD_OPTIMIZER_NAME:
        return optim.SGD(parameters, lr=config[LEARNING_RATE])

    if optimizer_name == ADAGRAD_OPTIMIZER_NAME:
        optim.Adagrad(parameters, lr=config[LEARNING_RATE])

    if optimizer_name == ADAM_OPTIMIZER_NAME:
        optim.Adam(parameters, lr=config[LEARNING_RATE])
