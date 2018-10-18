# -*- coding: utf-8 -*-

"""PyKEEN's command line interface."""

import json
import os
from collections import OrderedDict

import click
import pandas as pd

from pykeen.constants import (
    CONV_E_NAME, DISTMULT_NAME, ERMLP_NAME, FILTER_NEG_TRIPLES, HPO_MODE, OUTPUT_DIREC, PREFERRED_DEVICE, RESCAL_NAME,
    SE_NAME, TEST_FILE_ERROR_MSG, TEST_FILE_PROMPT_MSG, TEST_SET_PATH, TEST_SET_RATIO, TRAINING_FILE_ERROR_MSG,
    TRAINING_FILE_PROMPT_MSG, TRAINING_MODE, TRAINING_SET_PATH, TRANS_D_NAME, TRANS_E_NAME, TRANS_H_NAME, TRANS_R_NAME,
    UM_NAME,
    EXECUTION_MODE, HPO_ITERS_PRINT_MSG, HPO_ITERS_PROMPT_MSG, HPO_ITERS_ERROR_MSG, NUM_OF_HPO_ITERS)
from pykeen.run import run
from pykeen.utilities.cli_utils import (
    configure_distmult_training_pipeline, configure_ermlp_training_pipeline, configure_rescal_training_pipeline,
    configure_se_training_pipeline, configure_trans_d_training_pipeline, configure_trans_e_training_pipeline,
    configure_trans_h_training_pipeline, configure_trans_r_training_pipeline, configure_um_training_pipeline,
)
from pykeen.utilities.cli_utils.cli_print_msg_helper import (
    print_ask_for_evlauation_message, print_execution_mode_message, print_filter_negative_triples_message, print_intro,
    print_section_divider, print_test_ratio_message, print_test_set_message,
    print_training_set_message, print_welcome_message,
)
from pykeen.utilities.cli_utils.cli_query_helper import (
    ask_for_evaluation, ask_for_filtering_of_negatives, ask_for_test_set, get_input_path, query_output_directory,
    select_embedding_model, select_keen_execution_mode, select_preferred_device, select_ratio_for_test_set,
    select_integer_value)
from pykeen.utilities.cli_utils.conv_e_cli import configure_conv_e_training_pipeline
from pykeen.utilities.cli_utils.distmult_cli import configure_distmult_hpo_pipeline
from pykeen.utilities.cli_utils.ermlp_cli import configure_ermlp_hpo_pipeline
from pykeen.utilities.cli_utils.rescal_cli import configure_rescal_hpo_pipeline
from pykeen.utilities.cli_utils.structured_embedding_cli import configure_se_hpo_pipeline
from pykeen.utilities.cli_utils.trans_d_cli import configure_trans_d_hpo_pipeline
from pykeen.utilities.cli_utils.trans_e_cli import configure_trans_e_hpo_pipeline
from pykeen.utilities.cli_utils.trans_h_cli import configure_trans_h_hpo_pipeline
from pykeen.utilities.cli_utils.trans_r_cli import configure_trans_r_hpo_pipeline
from pykeen.utilities.cli_utils.unstructured_model_cli import configure_um_hpo_pipeline

MODEL_TRAINING_CONFIG_FUNCS = {
    TRANS_E_NAME: configure_trans_e_training_pipeline,
    TRANS_H_NAME: configure_trans_h_training_pipeline,
    TRANS_R_NAME: configure_trans_r_training_pipeline,
    TRANS_D_NAME: configure_trans_d_training_pipeline,
    SE_NAME: configure_se_training_pipeline,
    UM_NAME: configure_um_training_pipeline,
    DISTMULT_NAME: configure_distmult_training_pipeline,
    ERMLP_NAME: configure_ermlp_training_pipeline,
    RESCAL_NAME: configure_rescal_training_pipeline,
    CONV_E_NAME: None
}

MODEL_HPO_CONFIG_FUNCS = {
    TRANS_E_NAME: configure_trans_e_hpo_pipeline,
    TRANS_H_NAME: configure_trans_h_hpo_pipeline,
    TRANS_R_NAME: configure_trans_r_hpo_pipeline,
    TRANS_D_NAME: configure_trans_d_hpo_pipeline,
    SE_NAME: configure_se_hpo_pipeline,
    UM_NAME: configure_um_hpo_pipeline,
    DISTMULT_NAME: configure_distmult_hpo_pipeline,
    ERMLP_NAME: configure_ermlp_hpo_pipeline,
    RESCAL_NAME: configure_rescal_hpo_pipeline,
    CONV_E_NAME: configure_conv_e_training_pipeline
}


def _configure_training_pipeline(model_name):
    model_config_func = MODEL_TRAINING_CONFIG_FUNCS.get(model_name)
    if model_config_func is None:
        raise KeyError(f'invalid model given: {model_name}')
    config = model_config_func(model_name)
    if config is None:
        raise NotImplementedError(f'{model_name} has not yet been implemented')
    return config


def _configure_hpo_pipeline(model_name):
    model_config_func = MODEL_HPO_CONFIG_FUNCS.get(model_name)
    if model_config_func is None:
        raise KeyError(f'invalid model given: {model_name}')
    config = model_config_func(model_name)
    if config is None:
        raise NotImplementedError(f'{model_name} has not yet been implemented')
    return config


def _configure_evaluation_specific_parameters(pykeen_exec_mode):
    """

    :param config:
    :return:
    """
    config = OrderedDict()

    if pykeen_exec_mode == TRAINING_MODE:
        # Step 1: Ask whether to evaluate the model
        print_ask_for_evlauation_message()
        is_evaluation_mode = ask_for_evaluation()
        print_section_divider()
    else:
        is_evaluation_mode = True

    # Step 2: Specify test set, if is_evaluation_mode==True
    if is_evaluation_mode:
        print_test_set_message()
        provide_test_set = ask_for_test_set()
        print_section_divider()

        if provide_test_set:
            test_set_path = get_input_path(prompt_msg=TEST_FILE_PROMPT_MSG,
                                           error_msg=TEST_FILE_ERROR_MSG)
            config[TEST_SET_PATH] = test_set_path
        else:
            print_test_ratio_message()
            test_set_ratio = select_ratio_for_test_set()
            config[TEST_SET_RATIO] = test_set_ratio

        print_section_divider()

        # Ask whether to use filtered negative triples
        print_filter_negative_triples_message()
        filter_negative_triples = ask_for_filtering_of_negatives()
        config[FILTER_NEG_TRIPLES] = filter_negative_triples
        print_section_divider()

    return config


def welcome_prompt():
    """

    :return:
    """
    # Step: Welcome + Intro
    print_welcome_message()
    print_section_divider()
    print_intro()
    print_section_divider()


def training_file_prompt(config):
    """

    :param config:
    :return:
    """
    print_training_set_message()
    config[TRAINING_SET_PATH] = get_input_path(
        prompt_msg=TRAINING_FILE_PROMPT_MSG,
        error_msg=TRAINING_FILE_ERROR_MSG,
    )

    return config


def execution_mode_prompt(config):
    """

    :param config:
    :return:
    """
    print_execution_mode_message()
    pykeen_exec_mode = select_keen_execution_mode()
    config[EXECUTION_MODE] = pykeen_exec_mode
    print_section_divider()

    return config


def model_selection_prompt():
    """

    :return:
    """
    model_name = select_embedding_model()
    print_section_divider()
    return model_name


def execution_mode_specific_prompt(config, model_name):
    """

    :param config:
    :param model_name:
    :return:
    """
    pykeen_exec_mode = config[EXECUTION_MODE]
    if pykeen_exec_mode == TRAINING_MODE:
        config.update(_configure_training_pipeline(model_name))
    elif pykeen_exec_mode == HPO_MODE:
        config.update(_configure_hpo_pipeline(model_name))

        # Query number of HPO iterations
        hpo_iter = select_integer_value(
            print_msg=HPO_ITERS_PRINT_MSG,
            prompt_msg=HPO_ITERS_PROMPT_MSG,
            error_msg=HPO_ITERS_ERROR_MSG)
        config[NUM_OF_HPO_ITERS] = hpo_iter
        print_section_divider()
    return config


def device_prompt(config):
    """

    :param config:
    :return:
    """
    config[PREFERRED_DEVICE] = select_preferred_device()
    return config


def output_direc_prompt(config):
    """

    :param config:
    :return:
    """
    config[OUTPUT_DIREC] = query_output_directory()
    return config


def prompt_config():
    """

    :return:
    """
    config = OrderedDict()

    # Step 1: Welcome + Intro
    welcome_prompt()

    # Step 2: Ask for training file
    config = training_file_prompt(config=config)
    print_section_divider()

    # Step 3: Ask for execution mode
    config = execution_mode_prompt(config=config)
    print_section_divider()

    # Step 4: Ask for model
    model_name = model_selection_prompt()
    print_section_divider()

    # Step 5: Query parameters depending on the selected execution mode
    config = execution_mode_specific_prompt(config=config, model_name=model_name)
    print_section_divider()

    config.update(_configure_evaluation_specific_parameters(config[EXECUTION_MODE]))

    print_section_divider()

    # Step 7: Query device to train on
    config = device_prompt(config=config)
    print_section_divider()

    # Step 8: Define output directory
    config = output_direc_prompt(config=config)
    print_section_divider()

    return config


@click.command()
@click.option('-c', '--config', type=click.File(), help='A PyKEEN JSON configuration file')
def main(config):
    """PyKEEN: A software for training and evaluating knowledge graph embeddings."""

    if config is not None:
        config = json.load(config)
    else:
        config = prompt_config()

    run(config)


@click.command()
@click.option('-d', '--directory', type=click.Path(file_okay=False, dir_okay=True), default=os.getcwd())
@click.option('-o', '--output', type=click.File('w'))
def summarize(directory: str, output):
    """Summarize contents of training and evaluation"""
    r = []
    for subdirectory_name in os.listdir(directory):
        subdirectory = os.path.join(directory, subdirectory_name)
        if not os.path.isdir(subdirectory):
            continue
        configuration_path = os.path.join(subdirectory, 'configuration.json')
        if not os.path.exists(configuration_path):
            click.echo("missing configuration")
            continue
        with open(configuration_path) as file:
            configuration = json.load(file)
        evaluation_path = os.path.join(subdirectory, 'evaluation_summary.json')
        if not os.path.exists(evaluation_path):
            click.echo("missing evaluation summary")
            continue
        with open(evaluation_path) as file:
            evaluation = json.load(file)
        r.append(dict(**configuration, **evaluation))
    df = pd.DataFrame(r)
    df.to_csv(output, sep='\t')


if __name__ == '__main__':
    main()
