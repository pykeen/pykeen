# -*- coding: utf-8 -*-

"""Implementation the command line interface needed for TransD."""

from collections import OrderedDict

from pykeen.constants import MARGIN_LOSS_PRINT_MSG, MARGIN_LOSS_PROMPT_MSG, MARGIN_LOSS_ERROR_MSG, \
    NORM_SCORING_FUNCTION_PRINT_MSG, \
    NORM_SCORING_FUNCTION_PROMPT_MSG, NORM_SCORING_FUNCTION_ERROR_MSG, LEARNING_RATE_PRINT_MSG, \
    LEARNING_RATE_PROMPT_MSG, LEARNING_RATE_ERROR_MSG, BATCH_SIZE_PRINT_MSG, BATCH_SIZE_PROMPT_MSG, \
    BATCH_SIZE_ERROR_MSG, EPOCH_PRINT_MSG, EPOCH_PROMPT_MSG, EPOCH_ERROR_MSG, EMBEDDING_DIM, SCORING_FUNCTION_NORM, \
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, MARGIN_LOSS, \
    ENTITIES_EMBEDDING_DIMENSION_PRINT_MSG, ENTITIES_EMBEDDING_DIMENSION_PROMPT_MSG, \
    ENTITIES_EMBEDDING_DIMENSION_ERROR_MSG, RELATION_EMBEDDING_DIMENSION_PRINT_MSG, \
    RELATION_EMBEDDING_DIMENSION_PROMPT_MSG, RELATION_EMBEDDING_DIMENSION_ERROR_MSG, RELATION_EMBEDDING_DIM, \
    EMBEDDING_DIMENSIONS_PRINT_MSG, EMBEDDING_DIMENSIONS_PROMPT_MSG, EMBEDDING_DIMENSIONS_ERROR_MSG, \
    MARGIN_LOSSES_PRINT_MSG, MARGIN_LOSSES_PROMPT_MSG, MARGIN_LOSSES_ERROR_MSG, NORMS_SCORING_FUNCTION_PRINT_MSG, \
    NORMS_SCORING_FUNCTION_PROMPT_MSG, NORMS_SCORING_FUNCTION_ERROR_MSG, RELATION_EMBEDDING_DIMENSIONS_PRINT_MSG, \
    RELATION_EMBEDDING_DIMENSIONS_PROMPT_MSG, RELATION_EMBEDDING_DIMENSIONS_ERROR_MSG, LEARNING_RATES_PRINT_MSG, \
    LEARNING_RATES_PROMPT_MSG, LEARNING_RATES_ERROR_MSG, BATCH_SIZES_PRINT_MSG, BATCH_SIZES_PROMPT_MSG, \
    BATCH_SIZES_ERROR_MSG, EPOCHS_PRINT_MSG, EPOCHS_PROMPT_MSG, EPOCHS_ERROR_MSG
from pykeen.utilities.cli_utils.cli_print_msg_helper import print_training_margin_loss_message, \
    print_scoring_fct_message, \
    print_section_divider, print_learning_rate_message, print_batch_size_message, \
    print_number_epochs_message, print_entities_embedding_dimension_message, \
    print_relations_embedding_dimension_message, print_hpo_embedding_dimensions_message, \
    print_hpo_margin_losses_message, print_hpo_scoring_fcts_message, print_hpo_learning_rates_message, \
    print_hpo_batch_sizes_message, print_hpo_epochs_message
from pykeen.utilities.cli_utils.cli_query_helper import select_integer_value, select_float_value, \
    select_positive_integer_values, select_float_values
from pykeen.utilities.cli_utils.utils import get_config_dict


def configure_trans_d_training_pipeline(model_name):
    """Configure Trans D from pipeline.

    :param str model_name: name of the model
    :rtype: OrderedDict
    :return: configuration dictionary
    """
    config = get_config_dict(model_name)

    # Step 1: Query embedding dimension for entities
    print_entities_embedding_dimension_message()
    embedding_dimension = select_integer_value(
        print_msg=ENTITIES_EMBEDDING_DIMENSION_PRINT_MSG,
        prompt_msg=ENTITIES_EMBEDDING_DIMENSION_PROMPT_MSG,
        error_msg=ENTITIES_EMBEDDING_DIMENSION_ERROR_MSG
    )
    config[EMBEDDING_DIM] = embedding_dimension
    print_section_divider()

    # Step 2: Query embedding dimension for relations
    print_relations_embedding_dimension_message()
    relation_embedding_dimension = select_integer_value(
        print_msg=RELATION_EMBEDDING_DIMENSION_PRINT_MSG,
        prompt_msg=RELATION_EMBEDDING_DIMENSION_PROMPT_MSG,
        error_msg=RELATION_EMBEDDING_DIMENSION_ERROR_MSG
    )
    config[RELATION_EMBEDDING_DIM] = relation_embedding_dimension
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

    # Step 3: Query L_p norm as scoring function
    print_scoring_fct_message()
    scoring_fct_norm = select_integer_value(print_msg=NORM_SCORING_FUNCTION_PRINT_MSG,
                                            prompt_msg=NORM_SCORING_FUNCTION_PROMPT_MSG,
                                            error_msg=NORM_SCORING_FUNCTION_ERROR_MSG)
    config[SCORING_FUNCTION_NORM] = scoring_fct_norm
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


def configure_trans_d_hpo_pipeline(model_name):
    config = get_config_dict(model_name)

    # Step 1: Query embedding dimensions for entities
    print_hpo_embedding_dimensions_message()
    entity_embedding_dimensions = select_positive_integer_values(EMBEDDING_DIMENSIONS_PRINT_MSG,
                                                          EMBEDDING_DIMENSIONS_PROMPT_MSG,
                                                          EMBEDDING_DIMENSIONS_ERROR_MSG)
    config[EMBEDDING_DIM] = entity_embedding_dimensions
    print_section_divider()

    # Step 2: Query embedding dimensions for relations
    print_relations_embedding_dimension_message()
    relation_embedding_dimensions = select_positive_integer_values(
        print_msg=RELATION_EMBEDDING_DIMENSIONS_PRINT_MSG,
        prompt_msg=RELATION_EMBEDDING_DIMENSIONS_PROMPT_MSG,
        error_msg=RELATION_EMBEDDING_DIMENSIONS_ERROR_MSG
    )
    config[RELATION_EMBEDDING_DIM] = relation_embedding_dimensions
    print_section_divider()

    # Step 3: Query margin losses
    print_hpo_margin_losses_message()
    magin_loss = select_float_values(
        print_msg=MARGIN_LOSSES_PRINT_MSG,
        prompt_msg=MARGIN_LOSSES_PROMPT_MSG,
        error_msg=MARGIN_LOSSES_ERROR_MSG)
    config[MARGIN_LOSS] = magin_loss
    print_section_divider()

    # Step 4: Query L_p norms to use as scoring function
    print_hpo_scoring_fcts_message()
    scoring_fct_norm = select_positive_integer_values(
        print_msg=NORMS_SCORING_FUNCTION_PRINT_MSG,
        prompt_msg=NORMS_SCORING_FUNCTION_PROMPT_MSG,
        error_msg=NORMS_SCORING_FUNCTION_ERROR_MSG
    )
    config[SCORING_FUNCTION_NORM] = scoring_fct_norm
    print_section_divider()

    # Step 5: Query learning rate
    print_hpo_learning_rates_message()
    learning_rate = select_float_values(
        print_msg=LEARNING_RATES_PRINT_MSG,
        prompt_msg=LEARNING_RATES_PROMPT_MSG,
        error_msg=LEARNING_RATES_ERROR_MSG)
    config[LEARNING_RATE] = learning_rate
    print_section_divider()

    # Step 6: Query batch size
    print_hpo_batch_sizes_message()
    batch_size = select_positive_integer_values(
        print_msg=BATCH_SIZES_PRINT_MSG,
        prompt_msg=BATCH_SIZES_PROMPT_MSG,
        error_msg=BATCH_SIZES_ERROR_MSG)
    config[BATCH_SIZE] = batch_size
    print_section_divider()

    # Step 7: Query number of epochs
    print_hpo_epochs_message()
    number_epochs = select_positive_integer_values(
        print_msg=EPOCHS_PRINT_MSG,
        prompt_msg=EPOCHS_PROMPT_MSG,
        error_msg=EPOCHS_ERROR_MSG)
    config[NUM_EPOCHS] = number_epochs
    print_section_divider()

    return config


