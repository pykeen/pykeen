# -*- coding: utf-8 -*-

"""pykeen's command line interface."""

import json
import os
from collections import OrderedDict

import click
from prompt_toolkit import prompt

from pykeen.constants import *
from pykeen.run import run

mapping = {'yes': True, 'no': False}
id_to_embedding_models = {1: 'TransE', 2: 'TransH', 3: 'TransR', 4: 'TransD', 5: 'RotE', 6: 'ConvE'}
embedding_models_to_ids = {value: key for key, value in id_to_embedding_models.items()}
metrics_maping = {1: 'mean_rank', 2: 'hits@k'}
normalization_mapping = {1: 'l1', 2: 'l2'}
execution_mode_mapping = {1: TRAINING_MODE, 2: HPO_MODE}
device_question_mapping = {'yes': GPU, 'no': CPU}


def print_welcome_message():
    print('#########################################################')
    print('#\t\t\t\t\t\t\t#')
    print('#\t\t Welcome to pykeen\t\t\t#')
    print('#\t\t\t\t\t\t\t#')
    print('#########################################################')
    print()


def select_execution_mode():
    print('Please select the mode in which pykeen should be started:')
    print('Training: 1')
    print('Hyper-parameter search: 2')
    is_valid_input = False

    while is_valid_input is False:
        user_input = prompt('> Please select one of the options: ')

        if user_input != '1' and user_input != '2':
            print("Invalid input, please type \'1\' or \'2\' to chose one of the provided execution modes")
        else:
            return int(user_input)


def select_embedding_model():
    print('Please select the embedding model you want to use:')
    print("TransE: 1")
    print("TransH: 2")
    print("TransR: 3")
    print("TransD: 4")
    print("ConvE: 6")
    is_valid_input = False

    while is_valid_input is False:
        user_input = prompt('> Please select one of the options: ')

        if user_input not in ['1', '2', '3', '4', '5', '6']:
            print(
                "Invalid input, please type a number between \'1\' and \'6\' for choosing one of the embedding models")
        else:
            is_valid_input = True
            user_input = int(user_input)

    return user_input


def select_positive_integer_values(print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False
    integers = []

    while is_valid_input is False:
        is_valid_input = True
        user_input = prompt(prompt_msg)
        user_input = user_input.split(',')

        for integer in user_input:
            if integer.isnumeric():
                integers.append(int(integer))
            else:
                print(error_msg)
                is_valid_input = False
                break

    return integers


def select_float_values(print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False
    float_values = []

    while is_valid_input is False:
        user_input = prompt(prompt_msg)
        user_input = user_input.split(',')

        for float_value in user_input:
            try:
                float_value = float(float_value)
                float_values.append(float_value)
            except ValueError:
                print(error_msg)
                break

        is_valid_input = True

    return float_values


def select_eval_metrics():
    print('Please select the evaluation metrics you want to use:')
    print("Mean rank: 1")
    print("Hits@k: 2")

    metrics = []

    is_valid_input = False

    while not is_valid_input:
        is_valid_input = True
        user_input = prompt('> Please select the options comma separated:')
        user_input = user_input.split(',')

        for choice in user_input:
            if choice == '1' or choice == '2':
                metrics.append(metrics_maping[int(choice)])
            else:
                print('Invalid input, please type in a sequence of integers (\'1\' and/or \'2\')')
                is_valid_input = False
                break

    metrics = list(set(metrics))

    return metrics


def _select_translational_based_hpo_params(selected_model):
    hpo_params = OrderedDict()
    embedding_dimensions = select_positive_integer_values(EMBEDDING_DIMENSION_PRINT_MSG,
                                                          EMBEDDING_DIMENSION_PROMPT_MSG,
                                                          EMBEDDING_DIMENSION_ERROR_MSG)
    hpo_params[EMBEDDING_DIM] = embedding_dimensions

    # ---------
    margin_losses = select_float_values(MARGIN_LOSSES_PRINT_MSG, MARGIN_LOSSES_PROMPT_MSG, MARGIN_LOSSES_ERROR_MSG)
    hpo_params[MARGIN_LOSS] = margin_losses

    if selected_model == TRANS_E_NAME:
        hpo_params[NORM_FOR_NORMALIZATION_OF_ENTITIES] = select_float_values(
            NORMS_FOR_NORMALIZATION_OF_ENTITIES_PRINT_MSG,
            NORMS_FOR_NORMALIZATION_OF_ENTITIES_PROMPT_MSG, NORMS_FOR_NORMALIZATION_OF_ENTITIES_ERROR_MSG)

    if selected_model == TRANS_E_NAME or selected_model == TRANS_H_NAME:
        print('----------------------------')

        hpo_params[SCORING_FUNCTION_NORM] = select_float_values(NORMS_SCROING_FUNCTION_PRINT_MSG,
                                                                NORMS_SCROING_FUNCTION_PROMPT_MSG,
                                                                NORMS_SCROING_FUNCTION_ERROR_MSG)
    if selected_model == TRANS_H_NAME:
        hpo_params[WEIGHT_SOFT_CONSTRAINT_TRANS_H] = select_float_values(WEIGHTS_SOFT_CONSTRAINT_TRANS_H_PRINT_MSG,
                                                                         WEIGHTS_SOFT_CONSTRAINT_TRANS_H_PROMPT_MSG,
                                                                         WEIGHTS_SOFT_CONSTRAINT_TRANS_H_ERROR_MSG)

    return hpo_params


def _select_height_and_width(embedding_dim):
    is_valid_input = False

    print(
        "Note: Height and width must be positive integers, and height * width must equal embedding dimension \'%d\'" % embedding_dim)

    while not is_valid_input:
        print("Select height for embedding dimension ", embedding_dim)
        height = prompt('> Height:')

        print("Select width for embedding dimension ", embedding_dim)
        width = prompt('> Width:')

        if not (height.isnumeric() and width.isnumeric() and int(height) * int(width) == embedding_dim):
            print("Invalid input. Height and width must be positive integers, and height * width must equal "
                  "embedding dimension \'%d\'" % embedding_dim)
        else:
            return int(height), int(width)


def select_heights_and_widths(embedding_dimensions):
    heights = []
    widths = []

    p_msg = 'Please select for each selected embedding dimension, corrpespoinding heights and widths.\n' \
            'Make sure that \'embedding dimension\' = \'height * width\' '

    for embedding_dim in embedding_dimensions:
        is_valid_input = False
        while not is_valid_input:
            print("Select height for embedding dimension ", embedding_dim)
            height = prompt('> Height:')

            print("Select width for embedding dimension ", embedding_dim)
            width = prompt('> Width:')

            if not (height.isnumeric() and width.isnumeric() and int(height) * int(width) == embedding_dim):
                print("Invalid input. Height and width must be positive integers, and height * width must equal "
                      "embedding dimension \'%d\'" % embedding_dim)
            else:
                heights.append(int(height))
                widths.append(int(width))
                is_valid_input = True

    return heights, widths


def _select_kernel_size(depending_param, print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False

    while not is_valid_input:
        kernel_param = prompt(prompt_msg % depending_param)

        if not (kernel_param.isnumeric() and int(kernel_param) <= depending_param):
            print(error_msg % depending_param)
        else:
            return int(kernel_param)


def select_kernel_sizes(depending_params, print_msg, prompt_msg, error_msg):
    kernel_params = []
    print(print_msg)

    for dep_param in depending_params:
        is_valid_input = False

        while not is_valid_input:
            kernel_param = prompt(prompt_msg % dep_param)

            if not (kernel_param.isnumeric() and int(kernel_param) <= dep_param):
                print(error_msg % dep_param)
            else:
                kernel_params.append(int(kernel_param))
                is_valid_input = True

    return kernel_params


def _select_conv_e_params():
    conv_e_params = OrderedDict()
    embedding_dim = select_integer_value(EMBEDDING_DIMENSION_PRINT_MSG, EMBEDDING_DIMENSION_PROMPT_MSG,
                                         EMBEDDING_DIMENSION_ERROR_MSG)
    height, width = _select_height_and_width(embedding_dim)
    num_input_channels = select_integer_value(CONV_E_INPUT_CHANNEL_PRINT_MSG, CONV_E_INPUT_CHANNEL_PROMPT_MSG,
                                              CONV_E_INPUT_CHANNEL_ERROR_MSG)

    num_output_channels = select_integer_value(CONV_E_OUT_CHANNEL_PRINT_MSG, CONV_E_OUT_CHANNEL_PROMPT_MSG,
                                               CONV_E_OUT_CHANNEL_ERROR_MSG)

    kernel_height = _select_kernel_size(height, CONV_E_KERNEL_HEIGHT_PRINT_MSG, CONV_E_KERNEL_HEIGHT_PROMPT_MSG,
                                        CONV_E_KERNEL_HEIGHT_ERROR_MSG)
    kernel_width = _select_kernel_size(width, CONV_E_KERNEL_WIDTH_PRINT_MSG, CONV_E_KERNEL_WIDTH_PROMPT_MSG,
                                       CONV_E_KERNEL_WIDTH_ERROR_MSG)

    input_dropout = select_float_value(CONV_E_INPUT_DROPOUT_PRINT_MSG,
                                       CONV_E_INPUT_DROPOUT_PROMPT_MSG,
                                       CONV_E_INPUT_DROPOUT_ERROR_MSG)

    output_dropout = select_float_value(CONV_E_OUTPUT_DROPOUT_PRINT_MSG, CONV_E_OUTPUT_DROPOUT_PROMPT_MSG,
                                        CONV_E_OUTPUT_DROPOUT_ERROR_MSG)

    feature_map_dropout = select_float_value(CONV_E_FEATURE_MAP_DROPOUT_PRINT_MSG,
                                             CONV_E__FEATURE_MAP_DROPOUT_PROMPT_MSG,
                                             CONV_E_FEATURE_MAP_DROPOUT_ERROR_MSG)

    conv_e_params[EMBEDDING_DIM] = embedding_dim
    conv_e_params[CONV_E_HEIGHT] = height
    conv_e_params[CONV_E_WIDTH] = width
    conv_e_params[CONV_E_INPUT_CHANNELS] = num_input_channels
    conv_e_params[CONV_E_OUTPUT_CHANNELS] = num_output_channels
    conv_e_params[CONV_E_KERNEL_HEIGHT] = kernel_height
    conv_e_params[CONV_E_KERNEL_WIDTH] = kernel_width
    conv_e_params[CONV_E_INPUT_DROPOUT] = input_dropout
    conv_e_params[CONV_E_OUTPUT_DROPOUT] = output_dropout
    conv_e_params[CONV_E_FEATURE_MAP_DROPOUT] = feature_map_dropout

    return conv_e_params


def _select_conv_e_hpo_params():
    hpo_params = OrderedDict()

    embedding_dimensions = select_positive_integer_values(EMBEDDING_DIMENSION_PRINT_MSG,
                                                          EMBEDDING_DIMENSION_PROMPT_MSG,
                                                          EMBEDDING_DIMENSION_ERROR_MSG)

    heights, widths = select_heights_and_widths(embedding_dimensions)

    input_channels = select_positive_integer_values(CONV_E_HPO_INPUT_CHANNELS_PRINT_MSG,
                                                    CONV_E_HPO_INPUT_CHANNELS_PROMPT_MSG,
                                                    CONV_E_HPO_INPUT_CHANNELS_ERROR_MSG)

    output_channels = select_positive_integer_values(CONV_E_HPO_OUT_CHANNELS_PRINT_MSG,
                                                     CONV_E_HPO_OUT_CHANNELS_PROMPT_MSG,
                                                     CONV_E_HPO_OUT_CHANNELS_ERROR_MSG)

    kernel_heights = select_kernel_sizes(heights, CONV_E_HPO_KERNEL_HEIGHTS_PRINT_MSG,
                                         CONV_E_HPO_KERNEL_HEIGHTS_PROMPT_MSG,
                                         CONV_E_HPO_KERNEL_HEIGHTS_ERROR_MSG)
    kernel_widths = select_kernel_sizes(widths, CONV_E_HPO_KERNEL_WIDTHS_PRINT_MSG, CONV_E_HPO_KERNEL_WIDTHS_PROMPT_MSG,
                                        CONV_E_HPO_KERNEL_WIDTHS_ERROR_MSG)

    hpo_params[EMBEDDING_DIM] = embedding_dimensions
    hpo_params[CONV_E_HEIGHT] = heights
    hpo_params[CONV_E_WIDTH] = widths
    hpo_params[CONV_E_INPUT_CHANNELS] = input_channels
    hpo_params[CONV_E_OUTPUT_CHANNELS] = output_channels
    hpo_params[CONV_E_KERNEL_HEIGHT] = kernel_heights
    hpo_params[CONV_E_KERNEL_WIDTH] = kernel_widths
    hpo_params[CONV_E_INPUT_DROPOUT] = select_float_values(CONV_E_HPO_INPUT_DROPOUTS_PRINT_MSG,
                                                           CONV_E_HPO_INPUT_DROPOUTS_PROMPT_MSG,
                                                           CONV_E_HPO_INPUT_DROPOUTS_ERROR_MSG)

    hpo_params[CONV_E_OUTPUT_DROPOUT] = select_float_values(CONV_E_HPO_OUTPUT_DROPOUT_PRINT_MSG,
                                                            CONV_E_HPO_OUTPUT_DROPOUT_PROMPT_MSG,
                                                            CONV_E_HPO_OUTPUT_DROPOUT_ERROR_MSG)

    hpo_params[CONV_E_FEATURE_MAP_DROPOUT] = select_float_values(CONV_E_HPO_FEATURE_MAP_DROPOUT_PRINT_MSG,
                                                                 CONV_E_HPO_FEATURE_MAP_DROPOUT_PROMPT_MSG,
                                                                 CONV_E_HPO_FEATURE_MAP_DROPOUT_ERROR_MSG)

    return hpo_params


def select_hpo_params(model_id):
    hpo_params = OrderedDict()
    hpo_params[KG_EMBEDDING_MODEL] = id_to_embedding_models[model_id]
    selected_model = id_to_embedding_models[model_id]

    if selected_model in [TRANS_D_NAME, TRANS_E_NAME, TRANS_H_NAME, TRANS_R_NAME]:
        # Model is one of the the translational based models
        param_dict = _select_translational_based_hpo_params(selected_model)
        hpo_params.update(param_dict)
    elif selected_model == CONV_E_NAME:
        # ConvE
        param_dict = _select_conv_e_hpo_params()
        hpo_params.update(param_dict)
    elif model_id == 'Y':
        # TODO: RESCAL
        exit(0)
    elif model_id == 'Z':
        # TODO: COMPLEX
        exit(0)

    # General params
    # --------
    learning_rates = select_float_values(LEARNING_RATES_PRINT_MSG, LEARNING_RATES_PROMPT_MSG, LEARNING_RATES_ERROR_MSG)
    hpo_params[LEARNING_RATE] = learning_rates

    # --------------
    batch_sizes = select_positive_integer_values(BATCH_SIZES_PRINT_MSG, BATCH_SIZES_PROMPT_MSG, BATCH_SIZES_ERROR_MSG)
    hpo_params[BATCH_SIZE] = batch_sizes

    epochs = select_positive_integer_values(EPOCHS_PRINT_MSG, EPOCHS_PROMPT_MSG, EPOCHS_ERROR_MSG)
    hpo_params[NUM_EPOCHS] = epochs

    hpo_iter = select_integer_value(HPO_ITERS_PRINT_MSG, HPO_ITERS_PROMPT_MSG,
                                    HPO_ITERS_ERROR_MSG)
    hpo_params[NUM_OF_HPO_ITERS] = hpo_iter

    return hpo_params


def get_data_input_path(print_msg):
    print(print_msg)

    is_valid_input = False

    while is_valid_input is False:
        user_input = prompt('> Path:')

        if os.path.exists(os.path.dirname(user_input)):
            return user_input

        print('Path doesn\'t exist, please type in new path')


def select_ratio_for_test_set():
    print('Select the ratio of the training set used for test (e.g. 0.5):')
    is_valid_input = False

    while is_valid_input is False:
        user_input = prompt('> Ratio: ')

        try:
            ratio = float(user_input)
            if 0. < ratio < 1.:
                return ratio
        except ValueError:
            pass

        print('Invalid input, please type in a number > 0. and < 1.')


def is_test_set_provided():
    print('Do you provide a test set?')
    is_valid_input = False

    while is_valid_input is False:
        user_input = prompt('> \'yes\' or \'no\': ')

        if user_input != 'yes' and user_input != 'no':
            print('Invalid input, please type in \'yes\' or \'no\'')
        else:
            return mapping[user_input]


def select_preferred_device():
    print('Do you want to use a GPU if available?')
    is_valid_input = False

    while not is_valid_input:
        user_input = prompt('> \'yes\' or \'no\':')
        if user_input == 'yes' or user_input == 'no':
            return device_question_mapping[user_input]
        else:
            print('Invalid input, please type in \'yes\' or \'no\'')


def select_integer_value(print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False

    while not is_valid_input:
        user_input = prompt(prompt_msg)

        if user_input.isnumeric():
            return int(user_input)
        else:
            print(error_msg)


def select_norm(print_msg):
    print(print_msg)
    print('L1-Norm: 1')
    print('L2-Norm: 2')
    is_valid_input = False

    while not is_valid_input:
        is_valid_input = True
        user_input = prompt('> L-p Norm:')

        if user_input == '1' or user_input == '2':
            return int(user_input)
        else:
            print('Invalid input, please type in \'1\' or \'2\'')


def select_training_model_params(model_id):
    kg_model_params = OrderedDict()
    selected_model = id_to_embedding_models[model_id]
    kg_model_params[KG_EMBEDDING_MODEL] = selected_model

    if selected_model in [TRANS_D_NAME, TRANS_E_NAME, TRANS_H_NAME, TRANS_R_NAME]:
        embedding_dimension = select_integer_value(EMBEDDING_DIMENSION_PRINT_MSG, EMBEDDING_DIMENSION_PROMPT_MSG,
                                                   EMBEDDING_DIMENSION_ERROR_MSG)

        kg_model_params[EMBEDDING_DIM] = embedding_dimension
        kg_model_params[SCORING_FUNCTION_NORM] = select_norm(NORM_SCORING_FUNCTION_PRINT_MSG)

        if selected_model == TRANS_E_NAME:
            kg_model_params[NORM_FOR_NORMALIZATION_OF_ENTITIES] = select_norm(ENTITIES_NORMALIZATION_PRINT_MSG)

        if selected_model == TRANS_H_NAME:
            kg_model_params[WEIGHT_SOFT_CONSTRAINT_TRANS_H] = select_float_value(
                WEIGHT_SOFT_CONSTRAINT_TRANS_H_PRINT_MSG, WEIGHT_SOFT_CONSTRAINT_TRANS_H_PROMPT_MSG,
                WEIGHT_SOFT_CONSTRAINT_TRANS_H_ERROR_MSG)

            print('----------------------------')

        if selected_model == TRANS_R_NAME or selected_model == TRANS_D_NAME:
            relation_embedding_dim = select_integer_value(RELATION_EMBEDDING_DIMENSION_PRINT_MSG,
                                                          RELATION_EMBEDDING_DIMENSION_PROMPT_MSG,
                                                          EMBEDDING_DIMENSION_ERROR_MSG)
            kg_model_params[RELATION_EMBEDDING_DIM] = relation_embedding_dim

        kg_model_params[MARGIN_LOSS] = select_float_value(MARGIN_LOSS_PRINT_MSG, MARGIN_LOSS_PROMPT_MSG,
                                                          MARGIN_LOSS_ERROR_MSG)
    if selected_model == CONV_E_NAME:
        # ConvE
        kg_model_params.update(_select_conv_e_params())

    kg_model_params[LEARNING_RATE] = select_float_value(LEARNING_RATE_PRINT_MSG, LEARNING_RATE_PROMPT_MSG,
                                                        LEARNING_RATE_ERROR_MSG)

    kg_model_params[BATCH_SIZE] = select_integer_value(BATCH_SIZE_PRINT_MSG, BATCH_SIZE_PROMPT_MSG,
                                                       BATCH_SIZE_ERROR_MSG)

    kg_model_params[NUM_EPOCHS] = select_integer_value(EPOCH_PRINT_MSG, EPOCH_PROMPT_MSG, EPOCH_ERROR_MSG)

    return kg_model_params


def select_float_value(print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False

    while not is_valid_input:
        user_input = prompt(prompt_msg)
        try:
            float_value = float(user_input)
            return float_value
        except ValueError:
            print(error_msg)


def ask_for_existing_configuration():
    print('Do you provide an existing configuration dictionary?')
    is_valid_input = False

    while is_valid_input is False:
        user_input = prompt('> \'yes\' or \'no\':')
        if user_input == 'yes' or user_input == 'no':
            return mapping[user_input]
        else:
            print('Invalid input, type \'yes\' or \'no\'')


def load_config_file():
    is_valid_input = False
    config_file_path = get_data_input_path(print_msg=CONFIG_FILE_PROMPT_MSG)

    while is_valid_input is False:
        with open(config_file_path, 'rb') as f:
            try:
                data = json.load(f)
                assert type(data) == dict or type(data) == OrderedDict
                return data
            except:
                print('Invalid file, configuration file must be serialised dictionary (.json)')
                config_file_path = get_data_input_path(print_msg=CONFIG_FILE_PROMPT_MSG)


def ask_binary_question(print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False

    while is_valid_input is False:
        user_input = prompt(prompt_msg)
        if user_input == 'yes' or user_input == 'no':
            return mapping[user_input]
        else:
            print(error_msg)


def get_output_directory():
    print('Please type in the path to the output directory')
    is_valid_input = False

    while is_valid_input is False:
        user_input = prompt('> Path to output director:')
        if os.path.exists(os.path.dirname(user_input)):
            return user_input
        else:
            print('Invalid input, please type in the path to an existing directory.')


def start_cli():
    config = OrderedDict()

    print_welcome_message()
    configuration_exits = ask_for_existing_configuration()

    if configuration_exits:
        config = load_config_file()
        config[OUTPUT_DIREC] = get_output_directory()
        return config

    print('----------------------------')
    exec_mode = select_execution_mode()
    exec_mode = execution_mode_mapping[exec_mode]
    print('----------------------------')
    embedding_model_id = select_embedding_model()
    print('----------------------------')

    if exec_mode == HPO_MODE:
        hpo_params = select_hpo_params(model_id=embedding_model_id)
        config[HYPER_PARAMTER_OPTIMIZATION_PARAMS] = hpo_params
    else:
        kg_model_params = select_training_model_params(model_id=embedding_model_id)
        config[KG_EMBEDDING_MODEL] = kg_model_params

    print('----------------------------')

    config[TRAINING_SET_PATH] = get_data_input_path(print_msg=TRAINING_SET_PRINT_MSG)

    use_test_set = is_test_set_provided()

    if use_test_set:
        config[TEST_SET_PATH] = get_data_input_path(print_msg=TEST_FILE_PROMPT_MSG)
    else:
        config[TEST_SET_RATIO] = select_ratio_for_test_set()

    print('----------------------------')
    config[PREFERRED_DEVICE] = select_preferred_device()

    config[OUTPUT_DIREC] = get_output_directory()

    return config


@click.command()
def main():
    """pykeen: A software for training and evaluating knowledge graph embeddings."""
    config = start_cli()
    run(config)
