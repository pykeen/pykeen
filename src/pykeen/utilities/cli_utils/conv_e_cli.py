# -*- coding: utf-8 -*-

"""Implementation the command line interface needed for TransE."""

from pykeen.constants import EMBEDDING_DIMENSION_PRINT_MSG, EMBEDDING_DIMENSION_PROMPT_MSG, \
    EMBEDDING_DIMENSION_ERROR_MSG, \
    LEARNING_RATE_PRINT_MSG, \
    LEARNING_RATE_PROMPT_MSG, LEARNING_RATE_ERROR_MSG, BATCH_SIZE_PRINT_MSG, BATCH_SIZE_PROMPT_MSG, \
    BATCH_SIZE_ERROR_MSG, EPOCH_PRINT_MSG, EPOCH_PROMPT_MSG, EPOCH_ERROR_MSG, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, \
    NUM_EPOCHS, \
    CONV_E_INPUT_CHANNEL_PRINT_MSG, CONV_E_INPUT_CHANNEL_PROMPT_MSG, CONV_E_INPUT_CHANNEL_ERROR_MSG, \
    CONV_E_KERNEL_HEIGHT_PRINT_MSG, CONV_E_KERNEL_HEIGHT_PROMPT_MSG, CONV_E_KERNEL_HEIGHT_ERROR_MSG, \
    CONV_E_KERNEL_WIDTH_PRINT_MSG, CONV_E_KERNEL_WIDTH_PROMPT_MSG, CONV_E_KERNEL_WIDTH_ERROR_MSG, \
    CONV_E_INPUT_DROPOUT_PRINT_MSG, CONV_E_INPUT_DROPOUT_PROMPT_MSG, CONV_E_INPUT_DROPOUT_ERROR_MSG, \
    CONV_E_OUTPUT_DROPOUT_PRINT_MSG, CONV_E_OUTPUT_DROPOUT_PROMPT_MSG, CONV_E_OUTPUT_DROPOUT_ERROR_MSG, \
    CONV_E_FEATURE_MAP_DROPOUT_PRINT_MSG, CONV_E__FEATURE_MAP_DROPOUT_PROMPT_MSG, CONV_E_FEATURE_MAP_DROPOUT_ERROR_MSG, \
    CONV_E_INPUT_CHANNELS, CONV_E_KERNEL_HEIGHT, CONV_E_KERNEL_WIDTH, CONV_E_INPUT_DROPOUT, CONV_E_OUTPUT_DROPOUT, \
    CONV_E_FEATURE_MAP_DROPOUT, CONV_E_OUT_CHANNEL_PRINT_MSG, CONV_E_OUT_CHANNEL_PROMPT_MSG, \
    CONV_E_OUT_CHANNEL_ERROR_MSG, CONV_E_OUTPUT_CHANNELS, CONV_E_HEIGHT, CONV_E_WIDTH, EMBEDDING_DIMENSIONS_PRINT_MSG, \
    EMBEDDING_DIMENSIONS_PROMPT_MSG, EMBEDDING_DIMENSIONS_ERROR_MSG, CONV_E_HPO_INPUT_CHANNELS_PRINT_MSG, \
    CONV_E_HPO_INPUT_CHANNELS_PROMPT_MSG, CONV_E_HPO_INPUT_CHANNELS_ERROR_MSG, CONV_E_HPO_OUT_CHANNELS_PRINT_MSG, \
    CONV_E_HPO_OUT_CHANNELS_PROMPT_MSG, CONV_E_HPO_OUT_CHANNELS_ERROR_MSG, CONV_E_HPO_KERNEL_HEIGHTS_PRINT_MSG, \
    CONV_E_HPO_KERNEL_HEIGHTS_PROMPT_MSG, CONV_E_HPO_KERNEL_HEIGHTS_ERROR_MSG, CONV_E_HPO_INPUT_DROPOUTS_PRINT_MSG, \
    CONV_E_HPO_INPUT_DROPOUTS_PROMPT_MSG, CONV_E_HPO_INPUT_DROPOUTS_ERROR_MSG, CONV_E_HPO_OUTPUT_DROPOUT_PRINT_MSG, \
    CONV_E_HPO_OUTPUT_DROPOUT_PROMPT_MSG, CONV_E_HPO_OUTPUT_DROPOUT_ERROR_MSG, CONV_E_HPO_FEATURE_MAP_DROPOUT_PRINT_MSG, \
    CONV_E_HPO_FEATURE_MAP_DROPOUT_PROMPT_MSG, CONV_E_HPO_FEATURE_MAP_DROPOUT_ERROR_MSG, LEARNING_RATES_PRINT_MSG, \
    LEARNING_RATES_PROMPT_MSG, LEARNING_RATES_ERROR_MSG, BATCH_SIZES_PRINT_MSG, BATCH_SIZES_PROMPT_MSG, \
    BATCH_SIZES_ERROR_MSG, EPOCHS_PRINT_MSG, EPOCHS_PROMPT_MSG, EPOCHS_ERROR_MSG
from pykeen.utilities.cli_utils.cli_print_msg_helper import print_training_embedding_dimension_message, \
    print_embedding_dimension_info_message, print_section_divider, print_learning_rate_message, \
    print_batch_size_message, \
    print_number_epochs_message, print_conv_e_width_height_message, print_conv_input_channels_message, \
    print_conv_kernel_height_message, print_conv_kernel_width_message, print_hpo_input_dropout_message, \
    print_output_dropout_message, print_conv_e_output_channels_message, \
    print_hpo_embedding_dimensions_message, print_hpo_conv_e_width_height_message, \
    print_conv_e_hpo_kernel_height_message, print_conv_e_hpo_kernel_width_message, print_hpo_learning_rates_message, \
    print_hpo_batch_sizes_message, print_hpo_epochs_message, print_hpo_output_dropout_message, \
    print_hpo_feature_maps_dropouts_message, print_feature_map_dropout_message
from pykeen.utilities.cli_utils.cli_query_helper import select_integer_value, select_float_value, \
    query_height_and_width_for_conv_e, query_kernel_param, select_zero_one_float_value, select_positive_integer_values, \
    select_heights_and_widths, select_kernel_sizes, select_float_values, select_zero_one_range_float_values
from pykeen.utilities.cli_utils.utils import get_config_dict


def configure_conv_e_training_pipeline(model_name):
    """Configure ConvE.

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

    # Step 2: Query height and width
    print_conv_e_width_height_message()
    height, width = query_height_and_width_for_conv_e(embedding_dimension)
    config[CONV_E_HEIGHT] = height
    config[CONV_E_WIDTH] = width
    print_section_divider()

    # Step 3: Query number of input channels
    print_conv_input_channels_message()
    num_input_channels = select_integer_value(CONV_E_INPUT_CHANNEL_PRINT_MSG,
                                              CONV_E_INPUT_CHANNEL_PROMPT_MSG,
                                              CONV_E_INPUT_CHANNEL_ERROR_MSG)
    config[CONV_E_INPUT_CHANNELS] = num_input_channels
    print_section_divider()

    # Step 4: Query number of output channels
    print_conv_e_output_channels_message()
    num_output_channels = select_integer_value(CONV_E_OUT_CHANNEL_PRINT_MSG,
                                               CONV_E_OUT_CHANNEL_PROMPT_MSG,
                                               CONV_E_OUT_CHANNEL_ERROR_MSG)
    config[CONV_E_OUTPUT_CHANNELS] = num_output_channels
    print_section_divider()

    # Step 4: Query kernel height
    print_conv_kernel_height_message()
    kernel_height = query_kernel_param(depending_param=height,
                                       print_msg=CONV_E_KERNEL_HEIGHT_PRINT_MSG,
                                       prompt_msg=CONV_E_KERNEL_HEIGHT_PROMPT_MSG,
                                       error_msg=CONV_E_KERNEL_HEIGHT_ERROR_MSG)
    config[CONV_E_KERNEL_HEIGHT] = kernel_height
    print_section_divider()

    # Step 5: Query kernel width
    print_conv_kernel_width_message()
    kernel_width = query_kernel_param(depending_param=width,
                                      print_msg=CONV_E_KERNEL_WIDTH_PRINT_MSG,
                                      prompt_msg=CONV_E_KERNEL_WIDTH_PROMPT_MSG,
                                      error_msg=CONV_E_KERNEL_WIDTH_ERROR_MSG)
    config[CONV_E_KERNEL_WIDTH] = kernel_width
    print_section_divider()

    # Step 6: Query dropout for input layer
    print_hpo_input_dropout_message()
    input_dropout = select_zero_one_float_value(print_msg=CONV_E_INPUT_DROPOUT_PRINT_MSG,
                                                prompt_msg=CONV_E_INPUT_DROPOUT_PROMPT_MSG,
                                                error_msg=CONV_E_INPUT_DROPOUT_ERROR_MSG)
    config[CONV_E_INPUT_DROPOUT] = input_dropout
    print_section_divider()

    # Step 7: Query dropout for output layer
    print_output_dropout_message()
    output_dropout = select_zero_one_float_value(print_msg=CONV_E_OUTPUT_DROPOUT_PRINT_MSG,
                                                 prompt_msg=CONV_E_OUTPUT_DROPOUT_PROMPT_MSG,
                                                 error_msg=CONV_E_OUTPUT_DROPOUT_ERROR_MSG)
    config[CONV_E_OUTPUT_DROPOUT] = output_dropout
    print_section_divider()

    # Step 8: Query feature map dropout for output layer
    print_feature_map_dropout_message()
    feature_map_dropout = select_zero_one_float_value(print_msg=CONV_E_FEATURE_MAP_DROPOUT_PRINT_MSG,
                                                      prompt_msg=CONV_E__FEATURE_MAP_DROPOUT_PROMPT_MSG,
                                                      error_msg=CONV_E_FEATURE_MAP_DROPOUT_ERROR_MSG)
    config[CONV_E_FEATURE_MAP_DROPOUT] = feature_map_dropout
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


def configure_conv_e_hpo_pipeline(model_name):
    """Configure ConvE.

        :param str model_name: name of the model
        :rtype: OrderedDict
        :return: configuration dictionary
        """
    config = get_config_dict(model_name)

    # Step 1: Query embedding dimension
    print_hpo_embedding_dimensions_message()
    embedding_dimensions = select_positive_integer_values(EMBEDDING_DIMENSIONS_PRINT_MSG,
                                                          EMBEDDING_DIMENSIONS_PROMPT_MSG,
                                                          EMBEDDING_DIMENSIONS_ERROR_MSG)
    config[EMBEDDING_DIM] = embedding_dimensions
    print_section_divider()

    # Step 2: Query height and width
    print_hpo_conv_e_width_height_message()
    heights, widths = select_heights_and_widths(embedding_dimensions)
    config[CONV_E_HEIGHT] = heights
    config[CONV_E_WIDTH] = widths
    print_section_divider()

    # Step 3: Query number of input channels
    print_conv_input_channels_message()
    num_input_channels = select_positive_integer_values(CONV_E_HPO_INPUT_CHANNELS_PRINT_MSG,
                                                        CONV_E_HPO_INPUT_CHANNELS_PROMPT_MSG,
                                                        CONV_E_HPO_INPUT_CHANNELS_ERROR_MSG)
    config[CONV_E_INPUT_CHANNELS] = num_input_channels
    print_section_divider()

    # Step 4: Query number of output channels
    print_conv_e_output_channels_message()
    num_output_channels = select_positive_integer_values(CONV_E_HPO_OUT_CHANNELS_PRINT_MSG,
                                                         CONV_E_HPO_OUT_CHANNELS_PROMPT_MSG,
                                                         CONV_E_HPO_OUT_CHANNELS_ERROR_MSG)
    config[CONV_E_OUTPUT_CHANNELS] = num_output_channels
    print_section_divider()

    # Step 4: Query kernel height
    print_conv_e_hpo_kernel_height_message()
    kernel_heights = select_kernel_sizes(depending_params=heights,
                                         print_msg=CONV_E_HPO_KERNEL_HEIGHTS_PRINT_MSG,
                                         prompt_msg=CONV_E_HPO_KERNEL_HEIGHTS_PROMPT_MSG,
                                         error_msg=CONV_E_HPO_KERNEL_HEIGHTS_ERROR_MSG)
    config[CONV_E_KERNEL_HEIGHT] = kernel_heights
    print_section_divider()

    # Step 5: Query kernel width
    print_conv_e_hpo_kernel_width_message()
    kernel_widths = select_kernel_sizes(depending_params=widths,
                                       print_msg=CONV_E_KERNEL_WIDTH_PRINT_MSG,
                                       prompt_msg=CONV_E_KERNEL_WIDTH_PROMPT_MSG,
                                       error_msg=CONV_E_KERNEL_WIDTH_ERROR_MSG)
    config[CONV_E_KERNEL_WIDTH] = kernel_widths
    print_section_divider()

    # Step 6: Query dropout for input layer
    print_hpo_input_dropout_message()
    input_dropout = select_float_values(CONV_E_HPO_INPUT_DROPOUTS_PRINT_MSG,
                                        CONV_E_HPO_INPUT_DROPOUTS_PROMPT_MSG,
                                        CONV_E_HPO_INPUT_DROPOUTS_ERROR_MSG)
    config[CONV_E_INPUT_DROPOUT] = input_dropout
    print_section_divider()

    # Step 7: Query dropout for output layer
    print_hpo_output_dropout_message()
    output_dropouts = select_zero_one_range_float_values(print_msg=CONV_E_HPO_OUTPUT_DROPOUT_PRINT_MSG,
                                                         prompt_msg=CONV_E_HPO_OUTPUT_DROPOUT_PROMPT_MSG,
                                                         error_msg=CONV_E_HPO_OUTPUT_DROPOUT_ERROR_MSG)
    config[CONV_E_OUTPUT_DROPOUT] = output_dropouts
    print_section_divider()

    # Step 8: Query feature map dropout for output layer
    print_hpo_feature_maps_dropouts_message()
    feature_map_dropouts = select_zero_one_range_float_values(print_msg=CONV_E_HPO_FEATURE_MAP_DROPOUT_PRINT_MSG,
                                                              prompt_msg=CONV_E_HPO_FEATURE_MAP_DROPOUT_PROMPT_MSG,
                                                              error_msg=CONV_E_HPO_FEATURE_MAP_DROPOUT_ERROR_MSG)
    config[CONV_E_FEATURE_MAP_DROPOUT] = feature_map_dropouts
    print_section_divider()

    # Step 9: Query learning rate
    print_hpo_learning_rates_message()
    learning_rates = select_float_values(
        print_msg=LEARNING_RATES_PRINT_MSG,
        prompt_msg=LEARNING_RATES_PROMPT_MSG,
        error_msg=LEARNING_RATES_ERROR_MSG)
    config[LEARNING_RATE] = learning_rates
    print_section_divider()

    # Step 10: Query batch size
    print_hpo_batch_sizes_message()
    batch_size = select_positive_integer_values(
        print_msg=BATCH_SIZES_PRINT_MSG,
        prompt_msg=BATCH_SIZES_PROMPT_MSG,
        error_msg=BATCH_SIZES_ERROR_MSG)
    config[BATCH_SIZE] = batch_size
    print_section_divider()

    # Step 11: Query number of epochs
    print_hpo_epochs_message()
    number_epochs = select_positive_integer_values(
        print_msg=EPOCHS_PRINT_MSG,
        prompt_msg=EPOCHS_PROMPT_MSG,
        error_msg=EPOCHS_ERROR_MSG)
    config[NUM_EPOCHS] = number_epochs
    print_section_divider()

    return config
