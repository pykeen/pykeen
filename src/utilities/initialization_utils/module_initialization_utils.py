# -*- coding: utf-8 -*-

from evaluation_methods.mean_rank_evaluator import MeanRankEvaluator
from kg_embeddings_model.trans_e import TransE
from kg_embeddings_model.trans_h import TransH
from utilities.constants import CLASS_NAME, TRANS_E, TRANS_H, MEAN_RANK_EVALUATOR


def get_evaluator(config):
    class_name = config[CLASS_NAME]

    if class_name == MEAN_RANK_EVALUATOR:
        return MeanRankEvaluator()


def get_kg_embedding_model(config):
    """

    :param config:
    :return:
    """
    class_name = config[CLASS_NAME]

    if class_name == TRANS_E:
        return TransE(config=config)
    elif class_name == TRANS_H:
        return TransH(config=config)
