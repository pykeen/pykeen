# -*- coding: utf-8 -*-

"""Constants defined for POEM."""

from torch.optim import Adagrad, Adam, SGD

"""Device related constants"""

PREFERRED_DEVICE = 'preferred_device'
GPU = 'gpu'
CPU = 'cpu'

"""Configuration-related constants"""

EXECUTION_MODE = 'execution_mode'
TRAINING_MODE = 'training_mode'
HPO_MODE = 'HPO_mode'
MARGIN_LOSS = 'margin_mrl'
NUM_ENTITIES = 'num_entities'
NUM_RELATIONS = 'num_relations'
EMBEDDING_DIM = 'embedding_dim'
LEARNING_RATE = 'learning_rate'
KG_ASSUMPTION = 'KG_assumption'
OWA = 'open_world_assumption'
CWA = 'closed_world_assumption'
NUMERIC_LITERALS = 'numeric_literals'
BATCH_SIZE = 'batch_size'
NUM_EPOCHS = 'num_epochs'
SEED = 'random_seed'

"""Optimizer related constants"""
SGD_OPTIMIZER_NAME = SGD.__name__
ADAGRAD_OPTIMIZER_NAME = Adagrad.__name__
ADAM_OPTIMIZER_NAME = Adam.__name__

"""Evaluator related constants"""
EVALUATOR = 'evaluator'
RANK_BASED_EVALUATOR = 'ranked_based_evaluator'
FILTER_NEG_TRIPLES = 'filter_negative_triples'

"""Data"""

TRAINING_SET_PATH = 'training_set_path'
TEST_SET_PATH = 'test_set_path'
TEST_SET_RATIO = 'test_set_ratio'
PATH_TO_NUMERIC_LITERALS = 'path_to_numeric_literals'

"""Model names"""

KG_EMBEDDING_MODEL_NAME = 'kg_embedding_model_name'
DISTMULT_LITERAL_NAME_OWA = 'DistMultLiteral_OWA'
DISTMULT_LITERAL_NAME_CWA = 'DistMultLiteral_CWA'
COMPLEX_CWA_NAME = 'Complex_CWA'
COMPLEX_LITERAL_NAME_CWA = 'ComplexLiteral_CWA'
TRANS_E_NAME = 'TransE'

"""DistMult related constants"""

INPUT_DROPOUT = 'input_dropout'

"""Factory related constants"""
FACTORY_NAME = 'factory_name'
TRIPLES_FACTORY = 'triples_factory'
NUMERIC_LITERALS_FACTORY = 'numeric_literals_factory'

"""Training loops related constants"""
