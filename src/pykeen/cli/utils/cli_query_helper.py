# -*- coding: utf-8 -*-

"""Helper script to query parameters."""

import os

import click
from prompt_toolkit import prompt

from pykeen.cli.utils.constants import (
    ID_TO_KG_MODEL_MAPPING, ID_TO_OPTIMIZER_MAPPING, KG_MODEL_TO_ID_MAPPING, OPTIMIZER_TO_ID_MAPPING,
)
from pykeen.constants import CPU, GPU, HPO_MODE, IMPORTERS, PYKEEN, TRAINING_MODE, TRANS_E_NAME


def _is_correct_format(path: str):
    return (
        any(
            path.startswith(prefix)
            for prefix in IMPORTERS
        )
        or path.endswith('.tsv')
        or path.endswith('.nt')
    )


def get_input_path(prompt_msg: str) -> str:
    while True:
        user_input = prompt(prompt_msg, ).strip('"').strip("'")

        if _is_correct_format(path=user_input):
            return user_input

        click.secho(
            'Invalid data source, following data sources are supported:\nA string path to a .tsv file containing 3 '
            'columns corresponding to subject, predicate, and object.\nA string path to a .nt RDF file serialized in '
            'N-Triples format.\nA string NDEx network UUID prefixed by "ndex:" like in '
            'ndex:f93f402c-86d4-11e7-a10d-0ac135e8bacf\n',
            fg='red',
        )


def select_keen_execution_mode(lib_name=PYKEEN):
    r = click.confirm(
        f'Do you have hyper-parameters? If not, {lib_name} will be configured for hyper-parameter search.',
        default=False,
    )
    return TRAINING_MODE if r else HPO_MODE


def select_embedding_model() -> str:
    number_width = 1 + round(len(KG_MODEL_TO_ID_MAPPING) / 10)
    for model, model_id in KG_MODEL_TO_ID_MAPPING.items():
        click.echo(f'{model_id: >{number_width}}. {model}')
    click.echo()

    available_models, ids = zip(*KG_MODEL_TO_ID_MAPPING.items())

    while True:
        user_input = click.prompt(
            'Please select the embedding model you want to train',
            default=TRANS_E_NAME,
        )

        if user_input not in ids and user_input not in available_models:
            click.secho(
                f"Invalid input, please type in a number between 1 and {len(KG_MODEL_TO_ID_MAPPING)} indicating "
                f"the model id.\nFor example, type 1 to select the model {available_models[0]} and press enter",
                fg='red',
            )
            click.echo()
        elif user_input in available_models:
            return user_input
        else:
            return ID_TO_KG_MODEL_MAPPING[user_input]


def select_integer_value(print_msg, prompt_msg, error_msg, default=None):
    click.echo(print_msg)

    while True:
        try:
            return click.prompt(prompt_msg, type=int, default=default)
        except ValueError:
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


def select_ratio_for_test_set():
    while True:
        ratio = click.prompt('> Please select the ratio', default=0.2, type=float)

        try:
            if 0. < ratio < 1.:
                return ratio
        except ValueError:
            pass

        click.echo('Invalid input, the ratio should be 0.< ratio < 1. (e.g. 0.2).\nPlease try again.')


def select_preferred_device() -> str:
    click.secho("Current Step: Please specify the preferred device (GPU or CPU)", fg='blue')
    c = click.confirm('Do you want to try using the GPU? ', default=False)
    return GPU if c else CPU


def ask_for_filtering_of_negatives():
    return click.confirm('Do you want to filter out negative triples during evaluation of your model?')


def query_output_directory() -> str:
    default_output_directory = os.environ.get('PYKEEN_DEFAULT_OUTPUT_DIRECTORY')
    if default_output_directory is not None:
        os.makedirs(default_output_directory, exist_ok=True)
        return default_output_directory

    click.echo('Please provide the path to your output directory.\n\n')

    while True:
        user_input = os.path.expanduser(click.prompt('Path to output directory'))

        if os.path.exists(os.path.dirname(user_input)):
            return user_input

        try:
            os.makedirs(user_input, exist_ok=True)
        except FileExistsError:
            click.echo('Invalid input, please make sure that the path to the directory exists.\n'
                       'Please try again.')
        else:
            return user_input


def query_height_and_width_for_conv_e(embedding_dim):
    click.echo("Note: Height and width must be positive positive integers.\n"
               "Besides, height * width must equal to  embedding dimension \'%d\'" % embedding_dim)
    click.echo()

    while True:
        height = click.prompt('> Height')

        if not height.isnumeric():
            click.echo("Invalid input, please make sure that height is a positive integer.")
            continue

        width = click.prompt('> Width')

        if not width.isnumeric():
            click.echo("Invalid input, please make sure that height is a positive integer.")
            continue

        if not (int(height) * int(width) == embedding_dim):
            click.echo("Invalid input, height * width are not equal to \'%d\' (your specified embedding dimension).\n"
                       "Please try again, and fulfill the constraint)" % embedding_dim)
        else:
            return int(height), int(width)


def query_kernel_param(depending_param, print_msg, prompt_msg, error_msg):
    click.echo(print_msg % depending_param)

    while True:
        kernel_param = prompt(prompt_msg)

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


def select_zero_one_range_float_values(print_msg, prompt_msg, error_msg):
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
            except ValueError:
                click.echo(error_msg)
                is_valid_input = False
                break

            if 0. <= float_value <= 1.:
                print("hey")
                float_values.append(float_value)
            else:
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
        user_input = [v.strip() for v in user_input.split(',')]
        is_valid_input = True

        for integer in user_input:
            if integer.isnumeric():
                integers.append(int(integer))
            else:
                click.echo(error_msg)
                is_valid_input = False
                break

    return integers


def select_optimizer():
    click.echo('Please select the optimizer you want to train your model with:')
    for optimizer, id in OPTIMIZER_TO_ID_MAPPING.items():
        click.echo("%s: %s" % (optimizer, id))

    ids = list(OPTIMIZER_TO_ID_MAPPING.values())
    available_optimizers = list(OPTIMIZER_TO_ID_MAPPING.keys())

    while True:
        user_input = prompt('> Please select one of the options: ')

        if user_input not in ids:
            click.echo(
                "Invalid input, please type in a number between %s and %s indicating the optimizer id.\n"
                "For example type %s to select the model %s and press enter" % (
                    available_optimizers[0], ids[0], ids[0], available_optimizers[0]))
            click.echo()
        else:
            return ID_TO_OPTIMIZER_MAPPING[user_input]


def select_heights_and_widths(embedding_dimensions):
    heights = []
    widths = []

    for embedding_dim in embedding_dimensions:
        is_valid_input = False
        while not is_valid_input:
            click.echo("Specify height for specified embedding dimension %d ." % embedding_dim)
            height = click.prompt('> Height', type=int)

            click.echo("Specify width for specified embedding dimension %d ." % embedding_dim)
            width = click.prompt('> Width', type=int)

            if not (0 < height and 0 < width and height * width == embedding_dim):
                click.echo("Invalid input - height and width must be positive integers and height * width must"
                           " equal the specified embedding dimension of \'%d\'." % embedding_dim)
            else:
                heights.append(height)
                widths.append(width)
                is_valid_input = True
        print()

    return heights, widths


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
        print()

    return kernel_params
