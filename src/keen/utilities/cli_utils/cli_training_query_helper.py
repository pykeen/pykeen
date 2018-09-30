# -*- coding: utf-8 -*-

'''Helper script to query parameters needed in training mode.'''

import os

from prompt_toolkit import prompt

from keen.constants import EXECUTION_MODE_MAPPING, KG_MODEL_TO_ID_MAPPING, ID_TO_KG_MODEL_MAPPING, \
    BINARY_QUESTION_MAPPING


def get_input_path(prompt_msg, error_msg):
    while True:
        user_input = prompt(prompt_msg)

        if os.path.exists(os.path.dirname(user_input)):
            return user_input

        print(error_msg)


def select_keen_execution_mode():
    print('If KEEN should be executed in training mode please type 1, and press enter, for hyper-parameter search,\n'
          'please type 2 and press enter.')
    print()
    print('Training: 1')
    print('Hyper-parameter search: 2')
    print()

    while True:
        user_input = prompt('> Please select one of the above mentioned options: ')

        if user_input != '1' and user_input != '2':
            print("Invalid input, please type \'1\' for training or \'2\' for hyper-parameter search.\n"
                  "Please try again.")
            print()
        else:
            user_input = int(user_input)
            return EXECUTION_MODE_MAPPING[user_input]


def select_embedding_model():
    print('Please select the embedding model you want to train:')
    for model, id in KG_MODEL_TO_ID_MAPPING.items():
        print("%s: %s" % (model, id))

    ids = list(KG_MODEL_TO_ID_MAPPING.values())
    available_models = list(KG_MODEL_TO_ID_MAPPING.keys())

    while True:
        user_input = prompt('> Please select one of the options: ')

        if user_input not in ids:
            print(
                "Invalid input, please type in a number between %s and %s indicating the model id.\n"
                "For example type %s to select the model %s and press enter" % (available_models[0], ids[0]))
            print()
        else:
            return ID_TO_KG_MODEL_MAPPING[user_input]

    return user_input


def select_integer_value(print_msg, prompt_msg, error_msg):
    print(print_msg)

    while True:
        user_input = prompt(prompt_msg)

        if user_input.isnumeric():
            return int(user_input)

        print(error_msg)


def select_float_value(print_msg, prompt_msg, error_msg):
    print(print_msg)

    while True:
        user_input = prompt(prompt_msg)
        try:
            float_value = float(user_input)
            return float_value
        except ValueError:
            print(error_msg)


def ask_for_evaluation():
    print('Do you want to evaluate your model?')

    while True:
        user_input = prompt('> Please type \'yes\' or \'no\': ')
        if user_input != 'yes' and user_input != 'no':
            print('Invalid input, please type \'yes\' or \'no\' and press enter.\n'
                  'If you type \'yes\' it means that you want to evaluate your model after it is trained.')
        else:
            return BINARY_QUESTION_MAPPING[user_input]

def ask_for_test_set():
    print('Do you provide a test set yourself?')

    while True:
        user_input = prompt('> Please type \'yes\' or \'no\': ')

        if user_input != 'yes' and user_input != 'no':
            print('Invalid input, please type \'yes\' or \'no\' and press enter.\n'
                  'If you type \'yes\' it means that you provide a test set yourself.')
        else:
            return BINARY_QUESTION_MAPPING[user_input]
