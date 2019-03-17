# -*- coding: utf-8 -*-

"""PyKEEN's command line interface."""

from collections import OrderedDict
from typing import Dict, Optional

import click

from pykeen.cli.dicts import MODEL_HPO_CONFIG_FUNCS, MODEL_TRAINING_CONFIG_FUNCS
from pykeen.cli.utils.cli_print_msg_helper import (
    print_ask_for_evlauation_message, print_filter_negative_triples_message, print_intro,
    print_model_selection_message, print_random_seed_message, print_section_divider, print_test_ratio_message,
    print_test_set_message, print_training_set_message, print_welcome_message,
)
from pykeen.cli.utils.cli_query_helper import (
    ask_for_filtering_of_negatives, get_input_path, query_output_directory, select_embedding_model,
    select_integer_value, select_keen_execution_mode, select_preferred_device, select_ratio_for_test_set,
)
from pykeen.constants import (
    EXECUTION_MODE, FILTER_NEG_TRIPLES, HPO_ITERS_ERROR_MSG, HPO_ITERS_PRINT_MSG, HPO_ITERS_PROMPT_MSG, HPO_MODE,
    NUM_OF_HPO_ITERS, OUTPUT_DIREC, PREFERRED_DEVICE, PYKEEN, SEED, SEED_ERROR_MSG, SEED_PRINT_MSG, SEED_PROMPT_MSG,
    TEST_FILE_PROMPT_MSG, TEST_SET_PATH, TEST_SET_RATIO, TRAINING_FILE_PROMPT_MSG, TRAINING_MODE, TRAINING_SET_PATH,
    VERSION,
)

__all__ = [
    'prompt_config',
]


def _configure_training_pipeline(model_name: str):
    model_config_func = MODEL_TRAINING_CONFIG_FUNCS.get(model_name)
    if model_config_func is None:
        raise KeyError(f'invalid model given: {model_name}')
    config = model_config_func(model_name)
    if config is None:
        raise NotImplementedError(f'{model_name} has not yet been implemented')
    return config


def _configure_hpo_pipeline(model_name: str):
    model_config_func = MODEL_HPO_CONFIG_FUNCS.get(model_name)
    if model_config_func is None:
        raise KeyError(f'invalid model given: {model_name}')
    config = model_config_func(model_name)
    if config is None:
        raise NotImplementedError(f'{model_name} has not yet been implemented')
    return config


def prompt_evaluation_parameters(config: Dict) -> None:
    """Prompt the user for evaluation parameters absed on the execution mode."""
    if config[EXECUTION_MODE] == TRAINING_MODE:
        # Step 1: Ask whether to evaluate the model
        print_ask_for_evlauation_message()
        is_evaluation_mode = click.confirm('Do you want to evaluate your model?')
        print_section_divider()
    else:
        is_evaluation_mode = True

    # Step 2: Specify test set, if is_evaluation_mode==True
    if is_evaluation_mode:
        print_test_set_message()
        provide_test_set = click.confirm('Do you provide a test set yourself?')
        print_section_divider()

        if provide_test_set:
            test_set_path = get_input_path(TEST_FILE_PROMPT_MSG)
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


def prompt_training_file(config: Dict) -> None:
    """Prompt the user for a training file."""
    print_training_set_message()
    config[TRAINING_SET_PATH] = get_input_path(TRAINING_FILE_PROMPT_MSG)


def prompt_execution_mode(config: Dict, lib_name: str = PYKEEN) -> None:
    """Prompt the user for the execution mode."""
    config[EXECUTION_MODE] = select_keen_execution_mode(lib_name=lib_name)


def prompt_embedding_model() -> str:
    """Prompt the user to select an embedding model."""
    print_model_selection_message()
    model_name = select_embedding_model()
    print_section_divider()
    return model_name


def prompt_execution_parameters(config: Dict, model_name: str) -> None:
    """Prompt the user for execution mode parameters."""
    pykeen_exec_mode = config[EXECUTION_MODE]

    if pykeen_exec_mode == TRAINING_MODE:
        config.update(_configure_training_pipeline(model_name))

    elif pykeen_exec_mode == HPO_MODE:
        config.update(_configure_hpo_pipeline(model_name))

        # Query number of HPO iterations
        hpo_iter = select_integer_value(
            print_msg=HPO_ITERS_PRINT_MSG,
            prompt_msg=HPO_ITERS_PROMPT_MSG,
            error_msg=HPO_ITERS_ERROR_MSG,
        )
        config[NUM_OF_HPO_ITERS] = hpo_iter
        print_section_divider()


def prompt_random_seed(config) -> None:
    """Query random seed."""
    print_random_seed_message()
    config[SEED] = select_integer_value(
        print_msg=SEED_PRINT_MSG,
        prompt_msg=SEED_PROMPT_MSG,
        error_msg=SEED_ERROR_MSG,
        default=0,
    )


def prompt_device(config: Dict) -> None:
    """Prompt the user for their preferred evaluation device."""
    config[PREFERRED_DEVICE] = select_preferred_device()


def prompt_config(*, config: Optional[Dict] = None, show_welcome: bool = True, do_prompt_training: bool = True) -> Dict:
    """Prompt the user for the run configuration."""
    if config is None:
        config = OrderedDict()

    config['pykeen-version'] = VERSION

    # Step 1: Welcome + Intro
    if show_welcome:
        print_welcome_message()
        print_section_divider()
        print_intro()
        print_section_divider()

    # Step 2: Ask for training file
    if do_prompt_training:
        prompt_training_file(config)
        print_section_divider()

    # Step 3: Ask for execution mode
    prompt_execution_mode(config)
    print_section_divider()

    # Step 4: Ask for model
    model_name = prompt_embedding_model()
    print_section_divider()

    # Step 5: Query parameters depending on the selected execution mode
    prompt_execution_parameters(config, model_name=model_name)
    print_section_divider()

    # Step 5.5: Prompt for evaluation parameters depending on the selected execution mode
    prompt_evaluation_parameters(config)

    # Step 6: Please select a random seed
    prompt_random_seed(config)
    print_section_divider()

    # Step 7: Query device to train on
    prompt_device(config)
    print_section_divider()

    # Step 8: Define output directory
    config[OUTPUT_DIREC] = query_output_directory()
    print_section_divider()

    return config
