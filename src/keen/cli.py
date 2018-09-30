# -*- coding: utf-8 -*-

'''KEEN's command line interface.'''

import click
from collections import OrderedDict

from keen.constants import TRAINING_FILE_PROMPT_MSG, TRAINING_FILE_ERROR_MSG, TRAINING_SET_PATH, TRAINING_MODE, \
    HPO_MODE, TRANS_E, TRANS_H, TRANS_R, TRANS_D, SE, UM, DISTMULT, ERMLP, RESCAL, CONV_E
from keen.utilities.cli_utils.cli_print_msg_helper import print_welcome_message, print_section_divider, print_intro, \
    print_training_set_message, print_execution_mode_message
from keen.utilities.cli_utils.cli_training_query_helper import get_input_path, select_keen_execution_mode, \
    select_embedding_model
from keen.utilities.cli_utils.trans_e_cli import configure_trans_e_training_pipeline


def _configure_training_pipeline(model_name):
    if model_name == TRANS_E:
        config = configure_trans_e_training_pipeline()
    elif model_name == TRANS_H:
        pass
    elif model_name == TRANS_R:
        pass
    elif model_name == TRANS_D:
        pass
    elif model_name == SE:
        pass
    elif model_name == UM:
        pass
    elif model_name == DISTMULT:
        pass
    elif model_name == ERMLP:
        pass
    elif model_name == RESCAL:
        pass
    elif model_name == CONV_E:
        pass

def _configure_hpo_pipeline(model_name):
    if model_name == TRANS_E:
        pass
    elif model_name == TRANS_H:
        pass
    elif model_name == TRANS_R:
        pass
    elif model_name == TRANS_D:
        pass
    elif model_name == SE:
        pass
    elif model_name == UM:
        pass
    elif model_name == DISTMULT:
        pass
    elif model_name == ERMLP:
        pass
    elif model_name == RESCAL:
        pass
    elif model_name == CONV_E:
        pass

def start_cli():
    config = OrderedDict()

    # Step 1: Welcome + Intro
    print_welcome_message()
    print_section_divider()
    print_intro()
    print_section_divider()

    # Step 2: Ask for existing configuration
    # TODO: Ask for an existing configuration


    # Step 3: Ask for training file
    print_training_set_message()
    path_to_training_data = get_input_path(prompt_msg=TRAINING_FILE_PROMPT_MSG, error_msg=TRAINING_FILE_ERROR_MSG)
    config[TRAINING_SET_PATH] = path_to_training_data
    print_section_divider()

    # Step 4: Ask for execution mode
    print_execution_mode_message()
    keen_exec_mode = select_keen_execution_mode()
    print_section_divider()

    # Step 5: Ask for model
    model_name = select_embedding_model()
    print_section_divider()

    # Step 6:
    if keen_exec_mode == TRAINING_MODE:
        _configure_training_pipeline(model_name)
    if keen_exec_mode == HPO_MODE:
        _configure_hpo_pipeline(model_name)

    return config

@click.command()
def main():
    """KEEN: A software for training and evaluating knowledge graph embeddings."""
    config = start_cli()
    # run(config)

if __name__ == '__main__':
    main()