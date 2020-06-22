# -*- coding: utf-8 -*-

"""Constants defined for PyKEEN."""

from typing import Callable, Dict

import numpy as np
from pkg_resources import iter_entry_points

VERSION = '0.0.27-dev'
EMOJI = '⚽️'

#: Functions for specifying exotic resources with a given prefix
IMPORTERS: Dict[str, Callable[[str], np.ndarray]] = {
    entry_point.name: entry_point.load()
    for entry_point in iter_entry_points(group='pykeen.data.importer')
}


def get_version() -> str:
    """Get the version."""
    return VERSION


PYKEEN = 'PyKEEN'

# KG embedding model
KG_EMBEDDING_MODEL_NAME = 'kg_embedding_model_name'
EXECUTION_MODE = 'execution_mode'

# Model names
SE_NAME = 'SE'
UM_NAME = 'UM'
TRANS_E_NAME = 'TransE'
TRANS_H_NAME = 'TransH'
TRANS_D_NAME = 'TransD'
TRANS_R_NAME = 'TransR'
DISTMULT_NAME = 'DistMult'
ERMLP_NAME = 'ERMLP'
CONV_E_NAME = 'ConvE'
RESCAL_NAME = 'RESCAL'

# Evaluator
EVALUATOR = 'evaluator'
MEAN_RANK_EVALUATOR = 'MeanRankEvaluator'
RANDOM_SEARCH_OPTIMIZER = 'random_search_optimizer'
EVAL_METRICS = 'eval_metrics'
MEAN_RANK = 'mean_rank'
HITS_AT_K = 'hits@k'
FILTER_NEG_TRIPLES = 'filter_negative_triples'

# Output paths
ENTITY_TO_EMBEDDINGS = 'entity_to_embeddings'
EVAL_RESULTS = 'eval_results'
ENTITY_TO_ID = 'entity_to_id'
RELATION_TO_ID = 'relation_to_id'

# Device related
PREFERRED_DEVICE = 'preferred_device'
CPU = 'cpu'
GPU = 'gpu'

# ML params
BATCH_SIZE = 'batch_size'
VOCAB_SIZE = 'vocab_size'
EMBEDDING_DIM = 'embedding_dim'
RELATION_EMBEDDING_DIM = 'relation_embedding_dim'
MARGIN_LOSS = 'margin_loss'
NUM_ENTITIES = 'num_entities'
NUM_RELATIONS = 'num_relations'
NUM_EPOCHS = 'num_epochs'
NUM_OF_HPO_ITERS = 'maximum_number_of_hpo_iters'
LEARNING_RATE = 'learning_rate'
TRAINING_MODE = 'Training_mode'
HPO_MODE = 'HPO_mode'
HYPER_PARAMTER_OPTIMIZATION_PARAMS = 'hyper_optimization_params'
TRAINING_SET_PATH = 'training_set_path'
TEST_SET_PATH = 'test_set_path'
TEST_SET_RATIO = 'test_set_ratio'
NORM_FOR_NORMALIZATION_OF_ENTITIES = 'normalization_of_entities'
SCORING_FUNCTION_NORM = 'scoring_function'
# TransH related
WEIGHT_SOFT_CONSTRAINT_TRANS_H = 'weighting_soft_constraint'
# ConvE related
CONV_E_INPUT_DROPOUT = 'conv_e_input_dropout'
CONV_E_OUTPUT_DROPOUT = 'conv_e_output_dropout'
CONV_E_FEATURE_MAP_DROPOUT = 'conv_e_feature_map_dropout'
CONV_E_HEIGHT = 'ConvE_height'
CONV_E_WIDTH = 'ConvE_width'
CONV_E_INPUT_CHANNELS = 'ConvE_input_channels'
CONV_E_OUTPUT_CHANNELS = 'ConvE_output_channels'
CONV_E_KERNEL_HEIGHT = 'ConvE_kernel_height'
CONV_E_KERNEL_WIDTH = 'ConvE_kernel_width'

# OPTIMIZER
OPTMIZER_NAME = 'optimizer'
SGD_OPTIMIZER_NAME = 'SGD'
ADAGRAD_OPTIMIZER_NAME = 'Adagrad'
ADAM_OPTIMIZER_NAME = 'Adam'

# Further Constants
SEED = 'random_seed'
OUTPUT_DIREC = 'output_direc'

# Pipeline outcome parameters
TRAINED_MODEL = 'trained_model'
LOSSES = 'losses'
ENTITY_TO_EMBEDDING = 'entity_to_embedding'
RELATION_TO_EMBEDDING = 'relation_to_embedding'
CONFIG = 'configuration'
FINAL_CONFIGURATION = 'final_configuration'
EVAL_SUMMARY = 'eval_summary'
# -----------------Command line interface messages-----------------

TRAINING_FILE_PROMPT_MSG = '> Please provide here the path to your training file: '
TRAINING_FILE_ERROR_MSG = 'An error occurred, either the path is not correct or the training file doesn\'t exist.\n' \
                          'Please try again.'

TEST_FILE_PROMPT_MSG = '> Please provide here the path to your test file: '
TEST_FILE_ERROR_MSG = 'An error occurred, either the path is not correct or the test file doesn\'t exist.\n' \
                      'Please try again.'

EMBEDDING_DIMENSION_PRINT_MSG = 'Please type the range of preferred embedding dimensions for entities comma-separated' \
                                ' (e.g. 50,100,200):'
EMBEDDING_DIMENSION_PROMPT_MSG = '> Please select the embedding dimensions:'
EMBEDDING_DIMENSION_ERROR_MSG = 'An error occurred, please positive integer as embedding dimension.'

ENTITIES_EMBEDDING_DIMENSION_PRINT_MSG = 'Please provide the embedding dimension for entities (e.g. 40). '
ENTITIES_EMBEDDING_DIMENSION_PROMPT_MSG = '> Entity embedding dimension: '
ENTITIES_EMBEDDING_DIMENSION_ERROR_MSG = 'An error occurred, please proive and positive integer as embedding dimension' \
                                         ' (e.g. 50).'

ENTITIES_EMBEDDING_DIMENSIONS_PRINT_MSG = 'Please provide (comma separated) the embedding dimension(s) for entities' \
                                          ' (e.g. 50, 100). '
ENTITIES_EMBEDDING_DIMENSIONS_PROMPT_MSG = '> Entity embedding dimensions: '
ENTITIES_EMBEDDING_DIMENSIONS_ERROR_MSG = 'An error occurred, the embedding dimensions must be positive integers and' \
                                          ' separated by a comma.\nPlease try again.'

BATCH_SIZES_PRINT_MSG = 'Please type (comma separated) the batch size(s) (e.g. 32, 64, 128):'
BATCH_SIZES_PROMPT_MSG = '> Batch size(s): '
BATCH_SIZES_ERROR_MSG = 'An error occurred, the batch sizes must be positive integers and separated by a comma.\n' \
                        'Please try again.'

EPOCHS_PRINT_MSG = 'Please type (comma separated) the number of epochs (e.g. 50, 100, 500).'
EPOCHS_PROMPT_MSG = '> Epochs: '
EPOCHS_ERROR_MSG = 'An error occurred, the number of epochs must be positive integers and separated by a comma.\n' \
                   'Please try again.'

LEARNING_RATES_PRINT_MSG = 'Please type (comma separated) the learning rate(s) (e.g. 0.1, 0.01, 0.0001).'
LEARNING_RATES_PROMPT_MSG = '> Learning rate(s): '
LEARNING_RATES_ERROR_MSG = 'An error occurred, the learning rates must be float values and separated by a comma.\n' \
                           'Please try again.'

MARGIN_LOSSES_PRINT_MSG = 'Please type (comma separated) the margin losses(s) comma separated (e.g. 1, 2, 10),' \
                          ' and press enter.'
MARGIN_LOSSES_PROMPT_MSG = '> Margin losse(s): '
MARGIN_LOSSES_ERROR_MSG = 'An error occurred, the margin losses must be positive float values and separated by a' \
                          ' comma (e.g. 0.5, 3)'

MARGIN_LOSS_PRINT_MSG = 'Please type in the margin losses:'
MARGIN_LOSS_PROMPT_MSG = '> Margin loss: '
MARGIN_LOSS_ERROR_MSG = 'An error occurred, please type in a float value.'

HPO_ITERS_PRINT_MSG = 'Please type (comma separated) the number of iterations of hyper-parameter search (e.g. 5)'
HPO_ITERS_PROMPT_MSG = '> Number of iterations'
HPO_ITERS_ERROR_MSG = 'An error occurred, please type in a positive integer for the maximum number of iterations.'

EMBEDDING_DIMENSION_PRINT_MSG = 'Please provide the embedding dimension of entities and relations, and press enter.'
EMBEDDING_DIMENSION_PROMPT_MSG = '> Embedding dimension: '
EMBEDDING_DIMENSION_ERROR_MSG = 'An error occurred, please provide a positive integer as embedding dimension (e.g. 20).'

EMBEDDING_DIMENSIONS_PRINT_MSG = 'Please provide (comma separated) the embedding dimension(s), and press enter.'
EMBEDDING_DIMENSIONS_PROMPT_MSG = '> Embedding dimensions: '
EMBEDDING_DIMENSIONS_ERROR_MSG = 'An error occurred, the embedding dimensions must be positive integers and separated\n' \
                                 'by a comma e.g. 50,100,200' \
                                 'Please try again. \n'

RELATION_EMBEDDING_DIMENSION_PRINT_MSG = 'Please provide the embedding dimension of relations:'
RELATION_EMBEDDING_DIMENSION_PROMPT_MSG = '> Relation embedding dimension: '
RELATION_EMBEDDING_DIMENSION_ERROR_MSG = 'An error occurred, please type in an integer as embedding dimension such' \
                                         ' as 30.'

RELATION_EMBEDDING_DIMENSIONS_PRINT_MSG = 'Please provide (comma separated) the embedding dimensions of relations:'
RELATION_EMBEDDING_DIMENSIONS_PROMPT_MSG = '> Relation embedding dimensions: '
RELATION_EMBEDDING_DIMENSIONS_ERROR_MSG = 'An error occurred, the relation embedding dimensions must be positive' \
                                          ' integers and separated\nby a comma e.g. 50, 100, 200. Please try again. \n'

LEARNING_RATE_PRINT_MSG = 'Please type in the learning rate.'
LEARNING_RATE_PROMPT_MSG = '> Learning rate: '
LEARNING_RATE_ERROR_MSG = 'An error occurred, the learning rate should be a positive float value.\n' \
                          'Please try again.'

BATCH_SIZE_PRINT_MSG = 'Please type the batch size comma:'
BATCH_SIZE_PROMPT_MSG = '> Batch size:'
BATCH_SIZE_ERROR_MSG = 'An error occurred, please select a integer.'

EPOCH_PRINT_MSG = 'Please type the number of epochs:'
EPOCH_PROMPT_MSG = 'Epochs'
EPOCH_ERROR_MSG = 'An error occurred, please select an integers.'

SEED_PRINT_MSG = 'Please specify the random seed.'
SEED_PROMPT_MSG = 'Random seed'
SEED_ERROR_MSG = 'An error occurred, please proive and positive integer as the random seed.'

ENTITIES_NORMALIZATION_PRINT_MSG = 'Please select a norm to use to normalize the entities. The norm should be a positive integer greater than 0'
ENTITIES_NORMALIZATION_PROMPT_MSG = '> Norm to use for normalization of the entities: '
ENTITIES_NORMALIZATION_ERROR_MSG = 'An error occurred, the norm should be an integer greater than 0, such as 1\n' \
                                   'Please try again.'

NORMS_FOR_NORMALIZATION_OF_ENTITIES_PRINT_MSG = 'Please select (comma separated) a list of norms to use for normalizing the entities.'
NORMS_FOR_NORMALIZATION_OF_ENTITIES_PROMPT_MSG = '> Norms: '
NORMS_FOR_NORMALIZATION_OF_ENTITIES_ERROR_MSG = 'An error occurred, the normalization norms should integers, greater than 0, \n' \
                                                ' and separated by a comma (e.g. 1, 2, 3). Please try again.'

NORM_SCORING_FUNCTION_PRINT_MSG = 'Please select a norm to use as a scoring function. The norm should be a positive integer greater than 0'
NORM_SCORING_FUNCTION_PROMPT_MSG = '> Norm to use as scoring function: '
NORM_SCORING_FUNCTION_ERROR_MSG = 'An error occurred, the norm for the scoring function should be an integer greater than 0, such as 1\n' \
                                  'Please try again.'

NORMS_SCORING_FUNCTION_PRINT_MSG = 'Please select (comma separated) a list of norms to use as a scoring functions. The norms should be a positive integers greater than 0'
NORMS_SCORING_FUNCTION_PROMPT_MSG = '> Norms to use as scoring function: '
NORMS_SCORING_FUNCTION_ERROR_MSG = 'An error occurred, the norms for the scoring functions should integers, greater than 0, \n' \
                                   ' and separated by a comma (e.g. 1, 2, 3). Please try again.'

SAVE_CONFIG_PRINT_MSG = 'Do you want to save the configuration file?'
SAVE_CONFIG_PROMPT_MSG = '> \'yes\' or \'no\':'
SAVE_CONFIG_ERROR_MSG = 'An error occurred, please type \'yes\' or \'no\'.'

K_FOR_HITS_AT_K_PRINT_MSG = 'Please select \'k\' for hits@k'
K_FOR_HITS_AT_K_PROMPT_MSG = '> k:'
K_FOR_HITS_AT_K_ERROR_MSG = 'An error occurred, \'k\' must be a positive integer.'

CONV_E_HPO_INPUT_CHANNELS_PRINT_MSG = 'Please select (comma separated) the number of input channels for ConvE'
CONV_E_HPO_INPUT_CHANNELS_PROMPT_MSG = '> Input channels:'
CONV_E_HPO_INPUT_CHANNELS_ERROR_MSG = 'An error occurred, input channels must be positive integers.'

CONV_E_INPUT_CHANNEL_PRINT_MSG = 'Please select the number of input channels.'
CONV_E_INPUT_CHANNEL_PROMPT_MSG = '> Number of input channels:'
CONV_E_INPUT_CHANNEL_ERROR_MSG = 'An error occurred, the number of input channels must be a positive integer.\n' \
                                 'Please try again.'

CONV_E_HPO_OUT_CHANNELS_PRINT_MSG = 'Please select (comma separated) the number of output channels for ConvE'
CONV_E_HPO_OUT_CHANNELS_PROMPT_MSG = '> Output channels:'
CONV_E_HPO_OUT_CHANNELS_ERROR_MSG = 'An error occurred, output channels must be positive integers.'

CONV_E_OUT_CHANNEL_PRINT_MSG = 'Please select the number of output channels.'
CONV_E_OUT_CHANNEL_PROMPT_MSG = '> Output channels:'
CONV_E_OUT_CHANNEL_ERROR_MSG = 'An error occurred, the number of output channels must be a positive number.'

CONV_E_HPO_KERNEL_HEIGHTS_PRINT_MSG = 'Please select the kernel heights for ConvE'
CONV_E_HPO_KERNEL_HEIGHTS_PROMPT_MSG = '> Kernel height for defined of height \'%d\':'
CONV_E_HPO_KERNEL_HEIGHTS_ERROR_MSG = 'An error occurred, kernel heights must be positive integers and <= than %d (defined height).'

CONV_E_KERNEL_HEIGHT_PRINT_MSG = 'Please select the height for the convolution kernel based on the specified embedding height of %d.'
CONV_E_KERNEL_HEIGHT_PROMPT_MSG = '> Convolution kernel: '
CONV_E_KERNEL_HEIGHT_ERROR_MSG = 'An error occurred, the kernel height must be a positive integer and <= than %d (defined height).'

CONV_E_HPO_KERNEL_WIDTHS_PRINT_MSG = 'Please select the wifth for the convolution kernel.'
CONV_E_HPO_KERNEL_WIDTHS_PROMPT_MSG = '> Kernel width for defined width %d: '
CONV_E_HPO_KERNEL_WIDTHS_ERROR_MSG = 'An error occurred, kernel widths mus be positive integers and <= than %d (defined width).'

CONV_E_KERNEL_WIDTH_PRINT_MSG = 'Please select the kernel width for ConvE based on the specified embedding width of %d'
CONV_E_KERNEL_WIDTH_PROMPT_MSG = '> Kernel width for defined width of \'%d\': '
CONV_E_KERNEL_WIDTH_ERROR_MSG = 'An error occurred, kernel width mus be a positive integer and <= than %d (defined width).'

TRAINING_SET_PRINT_MSG = 'Please provide the path to the training file.'
TRAINING_FILE_ERROR_MSG = 'An error occurred, either the path is not correct or the training file doesn\'t exist.\n' \
                          'Please try again.'

CONFIG_FILE_PROMPT_MSG = '> Please provide the path to your existing configuration file: '
CONFIG_FILE_ERROR_MSG = 'An error occurred, please make sure that the file exists, and that it is JSON file.\n' \
                        'Please try again.'

CONV_E_HPO_INPUT_DROPOUTS_PRINT_MSG = 'Please select (comma separated) the input dropout value(s)'
CONV_E_HPO_INPUT_DROPOUTS_PROMPT_MSG = '> Input dropout value(s): '
CONV_E_HPO_INPUT_DROPOUTS_ERROR_MSG = 'An error occurred, input must be positive float values.'

CONV_E_INPUT_DROPOUT_PRINT_MSG = 'Please select the dropout rate for the input layer.'
CONV_E_INPUT_DROPOUT_PROMPT_MSG = '> Dropout rate: '
CONV_E_INPUT_DROPOUT_ERROR_MSG = 'An error occurred, the dropout rate must be a positive float value between 0 and 1.'

CONV_E_HPO_OUTPUT_DROPOUT_PRINT_MSG = 'Please select (comma separated) the output dropout value(s)'
CONV_E_HPO_OUTPUT_DROPOUT_PROMPT_MSG = '> Output dropout value(s):'
CONV_E_HPO_OUTPUT_DROPOUT_ERROR_MSG = 'An error occurred, input must be positive float values.'

CONV_E_OUTPUT_DROPOUT_PRINT_MSG = 'Please select the output dropout value'
CONV_E_OUTPUT_DROPOUT_PROMPT_MSG = '> Output dropout value: '
CONV_E_OUTPUT_DROPOUT_ERROR_MSG = 'An error occurred, input must be positive float values.'

CONV_E_HPO_FEATURE_MAP_DROPOUT_PRINT_MSG = 'Please select (comma separated) the feature map dropout value(s)'
CONV_E_HPO_FEATURE_MAP_DROPOUT_PROMPT_MSG = '> Feature map dropout value(s): '
CONV_E_HPO_FEATURE_MAP_DROPOUT_ERROR_MSG = 'An error occurred, input must be positive float values.'

CONV_E_FEATURE_MAP_DROPOUT_PRINT_MSG = 'Please select the feature map dropout value'
CONV_E__FEATURE_MAP_DROPOUT_PROMPT_MSG = '> Feature map dropout value:'
CONV_E_FEATURE_MAP_DROPOUT_ERROR_MSG = 'Invalid output, input must be a positive float value.'

WEIGHT_SOFT_CONSTRAINT_TRANS_H_PRINT_MSG = 'Please select the weight value for the soft constraints of' \
                                           ' the loss function'
WEIGHT_SOFT_CONSTRAINT_TRANS_H_PROMPT_MSG = '> Weight value for soft constraints: '
WEIGHT_SOFT_CONSTRAINT_TRANS_H_ERROR_MSG = 'An error occurred, input must be positive a float value.'

WEIGHTS_SOFT_CONSTRAINT_TRANS_H_PRINT_MSG = 'Please provide (comma separated) the weight values for weighting the' \
                                            ' soft constraints of the loss function'
WEIGHTS_SOFT_CONSTRAINT_TRANS_H_PROMPT_MSG = '> Weight values for soft constraints: '
WEIGHTS_SOFT_CONSTRAINT_TRANS_H_ERROR_MSG = 'An error occurred, the weight values for the soft constraint must be' \
                                            ' float values, \n and separated by a comma (e.g. 0.1, 0.5).' \
                                            ' Please try again.'
