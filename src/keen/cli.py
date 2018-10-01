# -*- coding: utf-8 -*-

'''KEEN's command line interface.'''

import click
from collections import OrderedDict

from keen.constants import TRAINING_FILE_PROMPT_MSG, TRAINING_FILE_ERROR_MSG, TRAINING_SET_PATH, TRAINING_MODE, \
    HPO_MODE, TRANS_E_NAME, TRANS_H_NAME, TRANS_R_NAME, TRANS_D_NAME, SE_NAME, UM_NAME, DISTMULT_NAME, ERMLP_NAME, \
    RESCAL_NAME, CONV_E_NAME, PREFERRED_DEVICE, TEST_FILE_PROMPT_MSG, TEST_FILE_ERROR_MSG, TEST_SET_PATH, \
    TEST_SET_RATIO, FILTER_NEG_TRIPLES, CONFIG_FILE_PROMPT_MSG, CONFIG_FILE_ERROR_MSG, OUTPUT_DIREC
from keen.run import run
from keen.utilities.cli_utils.cli_print_msg_helper import print_welcome_message, print_section_divider, print_intro, \
    print_training_set_message, print_execution_mode_message, print_ask_for_evlauation_message, print_test_set_message, \
    print_test_ratio_message, print_filter_negative_triples_message, print_existing_config_message, print_output_directory_message
from keen.utilities.cli_utils.cli_training_query_helper import get_input_path, select_keen_execution_mode, \
    select_embedding_model, select_preferred_device, ask_for_evaluation, ask_for_test_set, select_ratio_for_test_set, \
    ask_for_filtering_of_negatives, load_config_file, ask_for_existing_config_file, query_output_directory
from keen.utilities.cli_utils.trans_d_cli import configure_trans_d_training_pipeline
from keen.utilities.cli_utils.trans_e_cli import configure_trans_e_training_pipeline
from keen.utilities.cli_utils.trans_h_cli import configure_trans_h_training_pipeline
from keen.utilities.cli_utils.trans_r_cli import configure_trans_r_training_pipeline


def _configure_training_pipeline(model_name):
    if model_name == TRANS_E_NAME:
        config = configure_trans_e_training_pipeline(model_name)

    elif model_name == TRANS_H_NAME:
        config = configure_trans_h_training_pipeline(model_name)

    elif model_name == TRANS_R_NAME:
        config = configure_trans_r_training_pipeline(model_name)

    elif model_name == TRANS_D_NAME:
        config = configure_trans_d_training_pipeline(model_name)

    elif model_name == SE_NAME:
        pass

    elif model_name == UM_NAME:
        pass

    elif model_name == DISTMULT_NAME:
        pass

    elif model_name == ERMLP_NAME:
        pass

    elif model_name == RESCAL_NAME:
        pass

    elif model_name == CONV_E_NAME:
        pass

    return config

def _configure_hpo_pipeline(model_name):
    if model_name == TRANS_E_NAME:
        pass
    elif model_name == TRANS_H_NAME:
        pass
    elif model_name == TRANS_R_NAME:
        pass
    elif model_name == TRANS_D_NAME:
        pass
    elif model_name == SE_NAME:
        pass
    elif model_name == UM_NAME:
        pass
    elif model_name == DISTMULT_NAME:
        pass
    elif model_name == ERMLP_NAME:
        pass
    elif model_name == RESCAL_NAME:
        pass
    elif model_name == CONV_E_NAME:
        pass

def _configure_training_specific_parameters():
    """

    :param config:
    :return:
    """

    config = OrderedDict()

    # Step 1: Ask whether to evaluate the model
    print_ask_for_evlauation_message()
    is_evaluation_mode = ask_for_evaluation()
    print_section_divider()

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

def start_cli():
    config = OrderedDict()

    # Step 1: Welcome + Intro
    print_welcome_message()
    print_section_divider()
    print_intro()
    print_section_divider()

    # Step 2: Ask for existing configuration
    print_existing_config_message()
    use_existing_config = ask_for_existing_config_file()

    if use_existing_config:
        return load_config_file()

    print_section_divider()


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

    # Step 6: Query parameters depending on the selected execution mode
    if keen_exec_mode == TRAINING_MODE:
        config.update(_configure_training_pipeline(model_name))
        config.update(_configure_training_specific_parameters())

    if keen_exec_mode == HPO_MODE:
        config.update(_configure_hpo_pipeline(model_name))

    print_section_divider()

    # Step 7: Query device to train on
    prefered_device = select_preferred_device()
    config[PREFERRED_DEVICE] = prefered_device

    # Step 8: Define output directory
    print_output_directory_message()
    out_put_direc = query_output_directory()
    config[OUTPUT_DIREC] = out_put_direc
    print_section_divider()

    return config

@click.command()
def main():
    """KEEN: A software for training and evaluating knowledge graph embeddings."""
    config = start_cli()
    run(config)

if __name__ == '__main__':
    main()