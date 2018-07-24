# -*- coding: utf-8 -*-

from corpus_reader.csqa_filtered_wikidata_reader import CSQAWikiDataReader
from corpus_reader.walking_rdf_and_owl_reader import WROCReader
from corpus_reader.wn18_reader import WN18Reader
from evaluation_methods.mean_rank_evaluator import MeanRankEvaluator
from kg_embeddings_model.trans_e import TransE
from kg_embeddings_model.trans_h import TransH
from utilities.constants import CLASS_NAME, WROC_READER, TRANS_E, TRANS_H, MEAN_RANK_EVALUATOR, CSQA_WIKIDATA_READER, \
    WN18_READER


def get_evaluator(config):
    class_name = config[CLASS_NAME]

    if class_name == MEAN_RANK_EVALUATOR:
        return MeanRankEvaluator()


def get_reader(config):
    """

    :param config:
    :return:
    """
    class_name = config[CLASS_NAME]

    if class_name == WROC_READER:
        return WROCReader()
    if class_name == CSQA_WIKIDATA_READER:
        return CSQAWikiDataReader()
    if class_name == WN18_READER:
        return WN18Reader()


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
