# -*- coding: utf-8 -*-

"""Implementation the command line interface needed for ERMLP."""

from pykeen.constants import EMBEDDING_DIMENSION_PRINT_MSG, EMBEDDING_DIMENSION_PROMPT_MSG, EMBEDDING_DIMENSION_ERROR_MSG, \
    MARGIN_LOSS_PRINT_MSG, MARGIN_LOSS_PROMPT_MSG, MARGIN_LOSS_ERROR_MSG, LEARNING_RATE_PRINT_MSG, \
    LEARNING_RATE_PROMPT_MSG, LEARNING_RATE_ERROR_MSG, BATCH_SIZE_PRINT_MSG, BATCH_SIZE_PROMPT_MSG, \
    BATCH_SIZE_ERROR_MSG, EPOCH_PRINT_MSG, EPOCH_PROMPT_MSG, EPOCH_ERROR_MSG, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, \
    NUM_EPOCHS, \
    MARGIN_LOSS
from pykeen.utilities.cli_utils.cli_print_msg_helper import print_training_embedding_dimension_message, \
    print_embedding_dimension_info_message, print_training_margin_loss_message, print_section_divider, \
    print_learning_rate_message, print_batch_size_message, \
    print_number_epochs_message
from pykeen.utilities.cli_utils.cli_training_query_helper import select_integer_value, select_float_value
from pykeen.utilities.cli_utils.utils import get_config_dict


def configure_ermlp_training_pipeline(model_name):
    """Configure ERMLP from pipeline.

    :param str model_name: name of the model
    :rtype: OrderedDict
    :return: configuration dictionary
    """
    config = get_config_dict(model_name)

    # Step 1: Query embedding dimension
    print_training_embedding_dimension_message()
    print_embedding_dimension_info_message()
    embedding_dimension = select_integer_value(
        print_msg=EMBEDDING_DIMENSION_PRINT_MSG,
        prompt_msg=EMBEDDING_DIMENSION_PROMPT_MSG,
        error_msg=EMBEDDING_DIMENSION_ERROR_MSG
    )
    config[EMBEDDING_DIM] = embedding_dimension
    print_section_divider()

    # Step 2: Query margin loss
    print_training_margin_loss_message()
    magin_loss = select_float_value(
        print_msg=MARGIN_LOSS_PRINT_MSG,
        prompt_msg=MARGIN_LOSS_PROMPT_MSG,
        error_msg=MARGIN_LOSS_ERROR_MSG
    )
    config[MARGIN_LOSS] = magin_loss
    print_section_divider()

    # Step 5: Query learning rate
    print_learning_rate_message()
    learning_rate = select_float_value(
        print_msg=LEARNING_RATE_PRINT_MSG,
        prompt_msg=LEARNING_RATE_PROMPT_MSG,
        error_msg=LEARNING_RATE_ERROR_MSG
    )
    config[LEARNING_RATE] = learning_rate
    print_section_divider()

    # Step 6: Query batch size
    print_batch_size_message()
    batch_size = select_integer_value(
        print_msg=BATCH_SIZE_PRINT_MSG,
        prompt_msg=BATCH_SIZE_PROMPT_MSG,
        error_msg=BATCH_SIZE_ERROR_MSG
    )
    config[BATCH_SIZE] = batch_size
    print_section_divider()

    # Step 7: Query number of epochs
    print_number_epochs_message()
    number_epochs = select_integer_value(
        print_msg=EPOCH_PRINT_MSG,
        prompt_msg=EPOCH_PROMPT_MSG,
        error_msg=EPOCH_ERROR_MSG
    )
    config[NUM_EPOCHS] = number_epochs
    print_section_divider()

    return config


def configure_ermlp_hpo_pipeline():
    pass
