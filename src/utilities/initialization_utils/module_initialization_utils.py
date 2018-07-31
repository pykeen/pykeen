# -*- coding: utf-8 -*-

from kg_embeddings_model.trans_e import TransE
from kg_embeddings_model.trans_h import TransH
from utilities.constants import TRANS_E, TRANS_H, KG_EMBEDDING_MODEL


def get_kg_embedding_model(config):
    """

    :param config:
    :return:
    """
    model_name = config[KG_EMBEDDING_MODEL]

    if model_name == TRANS_E:
        return TransE(config=config)
    elif model_name == TRANS_H:
        return TransH(config=config)
