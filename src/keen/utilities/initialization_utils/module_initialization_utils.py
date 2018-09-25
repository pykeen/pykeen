# -*- coding: utf-8 -*-

from keen.constants import *
from keen.kg_embeddings_model.conv_e import ConvE
from keen.kg_embeddings_model.rot_e import RotE
from keen.kg_embeddings_model.trans_d import TransD
from keen.kg_embeddings_model.trans_e import TransE
from keen.kg_embeddings_model.trans_h import TransH
from keen.kg_embeddings_model.trans_r import TransR


def get_kg_embedding_model(config):
    """

    :param config:
    :return:
    """
    model_name = config[KG_EMBEDDING_MODEL]

    if model_name == TRANS_E:
        return TransE(config=config)
    if model_name == TRANS_H:
        return TransH(config=config)
    if model_name == TRANS_D:
        return TransD(config=config)
    if model_name == TRANS_R:
        return TransR(config=config)
    if model_name == CONV_E:
        return ConvE(config=config)
    if model_name == ROT_E:
        return RotE(config=config)
