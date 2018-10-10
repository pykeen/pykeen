# -*- coding: utf-8 -*-

"""Helper script to query parameters needed in training mode."""

import json
import os
from collections import OrderedDict

import click
from prompt_toolkit import prompt

from pykeen.constants import EXECUTION_MODE_MAPPING, KG_MODEL_TO_ID_MAPPING, ID_TO_KG_MODEL_MAPPING, \
    BINARY_QUESTION_MAPPING, GPU, CPU, CONFIG_FILE_PROMPT_MSG, CONFIG_FILE_ERROR_MSG


def get_input_path(prompt_msg, error_msg):
    while True:
        user_input = prompt(prompt_msg,).strip('"').strip("'")

        if os.path.exists(os.path.dirname(user_input)):
            return user_input

        click.echo(error_msg)


def select_keen_execution_mode():
    click.echo('Training: 1')
    click.echo('Hyper-parameter search: 2')
    click.echo()

    while True:
        user_input = prompt('> Please select one of the above mentioned options: ')

        if user_input != '1' and user_input != '2':
            click.echo("Invalid input, please type \'1\' for training or \'2\' for hyper-parameter search.\n"
                       "Please try again.")
            click.echo()
        else:
            user_input = int(user_input)
            return EXECUTION_MODE_MAPPING[user_input]


def select_embedding_model():
    click.echo('Please select the embedding model you want to train:')
    for model, id in KG_MODEL_TO_ID_MAPPING.items():
        click.echo("%s: %s" % (model, id))

    ids = list(KG_MODEL_TO_ID_MAPPING.values())
    available_models = list(KG_MODEL_TO_ID_MAPPING.keys())

    while True:
        user_input = prompt('> Please select one of the options: ')

        if user_input not in ids:
            click.echo(
                "Invalid input, please type in a number between %s and %s indicating the model id.\n"
                "For example type %s to select the model %s and press enter" % (
                available_models[0], ids[0], ids[0], available_models[0]))
            click.echo()
        else:
            return ID_TO_KG_MODEL_MAPPING[user_input]


def select_integer_value(print_msg, prompt_msg, error_msg):
    click.echo(print_msg)

    while True:
        user_input = prompt(prompt_msg)

        if user_input.isnumeric():
            return int(user_input)

        click.echo(error_msg)


def select_float_value(print_msg, prompt_msg, error_msg):
    click.echo(print_msg)

    while True:
        user_input = prompt(prompt_msg)
        try:
            float_value = float(user_input)
            return float_value
        except ValueError:
            click.echo(error_msg)


def select_zero_one_float_value(print_msg, prompt_msg, error_msg):
    click.echo(print_msg)

    while True:
        user_input = prompt(prompt_msg)
        try:
            float_value = float(user_input)
            if not (0 <= float_value <= 1):
                continue
            return float_value
        except ValueError:
            click.echo(error_msg)


def ask_for_evaluation():
    click.echo('Do you want to evaluate your model?')

    while True:
        user_input = prompt('> Please type \'yes\' or \'no\': ')
        if user_input != 'yes' and user_input != 'no':
            click.echo('Invalid input, please type \'yes\' or \'no\' and press enter.\n'
                       'If you type \'yes\' it means that you want to evaluate your model after it is trained.')
        else:
            return BINARY_QUESTION_MAPPING[user_input]


def ask_for_test_set():
    click.echo('Do you provide a test set yourself?')

    while True:
        user_input = prompt('> Please type \'yes\' or \'no\': ')

        if user_input != 'yes' and user_input != 'no':
            click.echo('Invalid input, please type \'yes\' or \'no\' and press enter.\n'
                       'If you type \'yes\' it means that you provide a test set yourself.')
        else:
            return BINARY_QUESTION_MAPPING[user_input]


def select_ratio_for_test_set():
    while True:
        user_input = prompt('> Please select the ratio: ')

        try:
            ratio = float(user_input)
            if 0. < ratio < 1.:
                return ratio
        except ValueError:
            pass

        click.echo('Invalid input, the ratio should be 0.< ratio < 1. (e.g. 0.2).\n'
                   'Please try again.')


def select_preferred_device():
    click.echo('Please select the preferred device (GPU or CPU).')

    while True:
        user_input = prompt('> Please type \'GPU\' or \'CPU\': ').lower()
        if user_input == GPU or user_input == CPU:
            return user_input
        else:
            click.echo('Invalid input, please type in \'GPU\' or \'CPU\' and press enter.')


def ask_for_filtering_of_negatives():
    click.echo('Do you want to filter out negative triples during evaluation of your model?')

    while True:
        user_input = prompt('> Please type \'yes\' or \'no\': ')

        if user_input != 'yes' and user_input != 'no':
            click.echo('Invalid input, please type \'yes\' or \'no\' and press enter.\n'
                       'If you type \'yes\' it means that you provide a test set yourself.')
        else:
            return BINARY_QUESTION_MAPPING[user_input]


def load_config_file():
    path_to_config_file = get_input_path(prompt_msg=CONFIG_FILE_PROMPT_MSG, error_msg=CONFIG_FILE_ERROR_MSG)
    while True:
        with open(path_to_config_file, 'rb') as f:
            try:
                config = json.load(f)
                assert type(config) == dict or type(config) == OrderedDict
                return config
            except:
                click.echo('Invalid file, the configuration must be a JSON file.\n'
                           'Please try again.')
                path_to_config_file = get_input_path(prompt_msg=CONFIG_FILE_PROMPT_MSG, error_msg=CONFIG_FILE_ERROR_MSG)


def ask_for_existing_config_file():
    click.echo('Do you provide an existing configuration file?\n')

    while True:
        user_input = prompt('> Please type \'yes\' or \'no\': ')

        if user_input != 'yes' and user_input != 'no':
            click.echo('Invalid input, please type \'yes\' or \'no\' and press enter.\n'
                       'If you type \'yes\' it means that you provide a configuration file.')
        else:
            return BINARY_QUESTION_MAPPING[user_input]


def query_output_directory():
    click.echo('Please provide the path to your output directory.\n')
    click.echo()

    while True:
        user_input = prompt('> Path to output director:')
        if os.path.exists(os.path.dirname(user_input)):
            return user_input
        else:
            click.echo('Invalid input, please make sure that the path to the directory exists.\n'
                       'Please try again.')


def query_height_and_width_for_conv_e(embedding_dim):
    click.echo("Note: Height and width must be positive positive integers.\n"
               "Besides, height * width must equal to  embedding dimension \'%d\'" % embedding_dim)
    click.echo()

    while True:
        height = prompt('> Height:')

        if not height.isnumeric():
            click.echo("Invalid input, please make sure that height is a positive integer.")
            continue

        width = prompt('> Width:')

        if not width.isnumeric():
            click.echo("Invalid input, please make sure that height is a positive integer.")
            continue

        if not (int(height) * int(width) == embedding_dim):
            click.echo("Invalid input, height * width are not equal to \'%d\' (your provided embedding dimension.\n"
                       "Please try again, and fulfill the constraint)" % embedding_dim)
        else:
            return int(height), int(width)


def query_kernel_param(depending_param, print_msg, prompt_msg, error_msg):
    click.echo(print_msg)

    while True:
        kernel_param = prompt(prompt_msg % depending_param)

        if not (kernel_param.isnumeric() and int(kernel_param) <= depending_param):
            click.echo(error_msg % depending_param)
        else:
            return int(kernel_param)


def select_float_values(print_msg, prompt_msg, error_msg):
    click.echo(print_msg)
    float_values = []
    is_valid_input = False

    while not is_valid_input:
        user_input = prompt(prompt_msg)
        user_input = user_input.split(',')
        is_valid_input = True

        for float_value in user_input:
            try:
                float_value = float(float_value)
                float_values.append(float_value)
            except ValueError:
                click.echo(error_msg)
                is_valid_input = False
                break

    return float_values


def select_positive_integer_values(print_msg, prompt_msg, error_msg):
    click.echo(print_msg)
    integers = []
    is_valid_input = False

    while not is_valid_input:
        user_input = prompt(prompt_msg)
        user_input = user_input.split(',')
        is_valid_input = True

        for integer in user_input:
            if integer.isnumeric():
                integers.append(int(integer))
            else:
                click.echo(error_msg)
                is_valid_input = False
                break

    return integers
