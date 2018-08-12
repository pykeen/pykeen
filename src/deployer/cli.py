import os
import pickle
import sys

w_dir = os.path.dirname(os.getcwd())
sys.path.append(w_dir)
from collections import OrderedDict

from prompt_toolkit import prompt

from utilities.constants import PREFERRED_DEVICE, EMBEDDING_DIMENSION_PRINT_MSG, \
    EMBEDDING_DIMENSION_PROMPT_MSG, EMBEDDING_DIMENSION_ERROR_MSG, MARGIN_LOSSES_PRINT_MSG, MARGIN_LOSSES_PROMPT_MSG, \
    MARGIN_LOSSES_ERROR_MSG, LEARNING_RATES_PRINT_MSG, LEARNING_RATES_PROMPT_MSG, LEARNING_RATES_ERROR_MSG, \
    BATCH_SIZES_PRINT_MSG, BATCH_SIZES_PROMPT_MSG, BATCH_SIZES_ERROR_MSG, EPOCHS_PRINT_MSG, EPOCHS_PROMPT_MSG, \
    EPOCHS_ERROR_MSG, MAX_HPO_ITERS_PRINT_MSG, MAX_HPO_ITERS_PROMPT_MSG, MAX_HPO_ITERS_ERROR_MSG, TRAINING, \
    HYPER_PARAMTER_SEARCH, HYPER_PARAMTER_OPTIMIZATION_PARAMS, EMBEDDING_DIM, KG_EMBEDDING_MODEL, MARGIN_LOSS, \
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, NUM_OF_MAX_HPO_ITERS, EVAL_METRICS, TRAINING_SET_PATH, VALIDATION_SET_PATH, \
    VALIDATION_SET_RATIO, NORMALIZATION_OF_ENTITIES, MARGIN_LOSS_PRINT_MSG, MARGIN_LOSS_PROMPT_MSG, \
    MARGIN_LOSS_ERROR_MSG, LEARNING_RATE_PRINT_MSG, LEARNING_RATE_PROMPT_MSG, LEARNING_RATE_ERROR_MSG, \
    BATCH_SIZE_PRINT_MSG, BATCH_SIZE_PROMPT_MSG, BATCH_SIZE_ERROR_MSG, EPOCH_PRINT_MSG, EPOCH_PROMPT_MSG, \
    EPOCH_ERROR_MSG, OUTPUT_DIREC, HITS_AT_K, \
    K_FOR_HITS_AT_K_PRINT_MSG, K_FOR_HITS_AT_K_PROMPT_MSG, K_FOR_HITS_AT_K_ERROR_MSG, K_FOR_HITS_AT_K, \
    TRAINING_SET_PRINT_MSG, VALIDATION_SET_PRINT_MSG, CONFIG_FILE_PRINT_MSG, CONV_E_INPUT_CHANNELS_PRINT_MSG, \
    CONV_E_INPUT_CHANNELS_PROMPT_MSG, CONV_E_INPUT_CHANNELS_ERROR_MSG, CONV_E_OUT_CHANNELS_PRINT_MSG, \
    CONV_E_OUT_CHANNELS_PROMPT_MSG, CONV_E_OUT_CHANNELS_ERROR_MSG, CONV_E_KERNEL_HEIGHTS_PRINT_MSG, \
    CONV_E_KERNEL_HEIGHTS_PROMPT_MSG, CONV_E_KERNEL_HEIGHTS_ERROR_MSG, CONV_E_KERNEL_WIDTHS_PRINT_MSG, \
    CONV_E_KERNEL_WIDTHS_PROMPT_MSG, CONV_E_KERNEL_WIDTHS_ERROR_MSG
from utilities.pipeline import Pipeline

mapping = {'yes': True, 'no': False}
embedding_models_mapping = {1: 'TransE', 2: 'TransH', 3: 'TransR', 4: 'TransD', 5:'ConvE'}
metrics_maping = {1: 'mean_rank', 2: 'hits@k'}
normalization_mapping = {1: 'l1', 2: 'l2'}
execution_mode_mapping = {1: TRAINING, 2: HYPER_PARAMTER_SEARCH}


def print_welcome_message():
    print('#########################################################')
    print('#\t\t\t\t\t\t\t#')
    print('#\t\t Welcome to KEEN\t\t\t#')
    print('#\t\t\t\t\t\t\t#')
    print('#########################################################')
    print()


def select_execution_mode():
    print('Please select the mode in which KEEN should be started:')
    print('Training: 1')
    print('Hyper-parameter search: 2')
    is_valid_input = False

    while is_valid_input == False:
        user_input = prompt('> Please select one of the options: ')

        if user_input != '1' and user_input != '2':
            print("Invalid input, please type \'1\' or \'2\' to chose one of the provided execution modes")
        else:
            return int(user_input)


def select_embedding_model():
    print('Please select the embedding model you want to use:')
    print("TransE: 1")
    print("TransH: 2")
    print("TransR: 3")
    print("TransD: 4")
    print("ConvE: 5")
    is_valid_input = False

    while is_valid_input == False:
        user_input = prompt('> Please select one of the options: ')

        if user_input not in ['1', '2', '3', '4', '5']:
            print(
                "Invalid input, please type a number between \'1\' and \'4\' for choosing one of the embedding models")
        else:
            is_valid_input = True
            user_input = int(user_input)

    return user_input


def select_positive_integer_values(print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False
    integers = []

    while is_valid_input == False:
        is_valid_input = True
        user_input = prompt(prompt_msg)
        user_input = user_input.split(',')

        for integer in user_input:
            if integer.isnumeric():
                integers.append(int(integer))
            else:
                print(error_msg)
                is_valid_input = False
                break

    return integers


def select_float_values(print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False
    float_values = []

    while is_valid_input == False:
        user_input = prompt(prompt_msg)
        user_input = user_input.split(',')

        for float_value in user_input:
            try:
                float_value = float(float_value)
                float_values.append(int(float_value))
            except ValueError:
                print(error_msg)
                break

        is_valid_input = True

    return float_values


def select_eval_metrics():
    print('Please select the evaluation metrics you want to use:')
    print("Mean rank: 1")
    print("Hits@k: 2")

    metrics = []

    is_valid_input = False

    while not is_valid_input:
        is_valid_input = True
        user_input = prompt('> Please select the options comma separated:')
        user_input = user_input.split(',')

        for choice in user_input:
            if choice == '1' or choice == '2':
                metrics.append(metrics_maping[int(choice)])
            else:
                print('Invalid input, please type in a sequence of integers (\'1\' and/or \'2\')')
                is_valid_input = False
                break

    metrics = list(set(metrics))

    return metrics


def _select_trans_x_params(model_id):
    hpo_params = OrderedDict()
    embedding_dimensions = select_positive_integer_values(EMBEDDING_DIMENSION_PRINT_MSG,
                                                          EMBEDDING_DIMENSION_PROMPT_MSG,
                                                          EMBEDDING_DIMENSION_ERROR_MSG)
    hpo_params[EMBEDDING_DIM] = embedding_dimensions

    # ---------
    margin_losses = select_float_values(MARGIN_LOSSES_PRINT_MSG, MARGIN_LOSSES_PROMPT_MSG, MARGIN_LOSSES_ERROR_MSG)
    hpo_params[MARGIN_LOSS] = margin_losses

    if model_id == 1:
        hpo_params[NORMALIZATION_OF_ENTITIES] = select_entites_normalization()

    return hpo_params


def select_heights_and_widths(embedding_dimensions):
    heights = []
    widths = []

    p_msg = 'Please select for each selected embedding dimension, corrpespoinding heights and widths.\n' \
            'Make sure that \'embedding dimension\' = \'height * width\' '

    print(p_msg)

    for embedding_dim in embedding_dimensions:
        is_valid_input = False
        while not is_valid_input:
            print("Select height for embedding dimension ", embedding_dim)
            height = prompt('> Height:')

            print("Select width for embedding dimension ", embedding_dim)
            width = prompt('> Width:')

            if not (height.isnumeric() and width.isnumeric() and int(height) * int(width) == embedding_dim):
                print("Invalid input. Height and width must be positive integers, and height * width must equal "
                      "embedding dimension \'%d\'" % embedding_dim)
            else:
                heights.append(int(height))
                widths.append(int(width))
                is_valid_input = True

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

    return kernel_params


def _select_conv_e_params():
    hpo_params = OrderedDict()

    embedding_dimensions = select_positive_integer_values(EMBEDDING_DIMENSION_PRINT_MSG,
                                                          EMBEDDING_DIMENSION_PROMPT_MSG,
                                                          EMBEDDING_DIMENSION_ERROR_MSG)

    heights, widths = select_heights_and_widths(embedding_dimensions)

    input_channels = select_positive_integer_values(CONV_E_INPUT_CHANNELS_PRINT_MSG, CONV_E_INPUT_CHANNELS_PROMPT_MSG,
                                                    CONV_E_INPUT_CHANNELS_ERROR_MSG)

    output_channels = select_positive_integer_values(CONV_E_OUT_CHANNELS_PRINT_MSG, CONV_E_OUT_CHANNELS_PROMPT_MSG,
                                                     CONV_E_OUT_CHANNELS_ERROR_MSG)

    kernel_heights = select_kernel_sizes(heights, CONV_E_KERNEL_HEIGHTS_PRINT_MSG, CONV_E_KERNEL_HEIGHTS_PROMPT_MSG,
                                         CONV_E_KERNEL_HEIGHTS_ERROR_MSG)
    kernel_widths = select_kernel_sizes(widths, CONV_E_KERNEL_WIDTHS_PRINT_MSG, CONV_E_KERNEL_WIDTHS_PROMPT_MSG,
                                        CONV_E_KERNEL_WIDTHS_ERROR_MSG)

    hpo_params['ConvE_heights'] = heights
    hpo_params['ConvE_widths'] = widths
    hpo_params['ConvE_input_channels'] = input_channels
    hpo_params['ConvE_output_channels'] = output_channels
    hpo_params['ConvE_kernel_heights'] = kernel_heights
    hpo_params['ConvE_kernel_widths'] = kernel_widths

    return hpo_params


def select_hpo_params(model_id):
    hpo_params = OrderedDict()
    hpo_params[KG_EMBEDDING_MODEL] = embedding_models_mapping[model_id]

    if 1 <= model_id and model_id <= 4:
        # Model is one of the TransX versions
        param_dict = _select_trans_x_params(model_id)
        hpo_params.update(param_dict)
    elif model_id == 5:
        # ConvE
        param_dict = _select_conv_e_params()
    elif model_id == 'Y':
        # TODO: RESCAL
        exit(0)
    elif model_id == 'Z':
        # TODO: COMPLEX
        exit(0)

    # General params
    # --------
    learning_rates = select_float_values(LEARNING_RATES_PRINT_MSG, LEARNING_RATES_PROMPT_MSG, LEARNING_RATES_ERROR_MSG)
    hpo_params[LEARNING_RATE] = learning_rates

    # --------------
    batch_sizes = select_positive_integer_values(BATCH_SIZES_PRINT_MSG, BATCH_SIZES_PROMPT_MSG, BATCH_SIZES_ERROR_MSG)
    hpo_params[BATCH_SIZE] = batch_sizes

    epochs = select_positive_integer_values(EPOCHS_PRINT_MSG, EPOCHS_PROMPT_MSG, EPOCHS_ERROR_MSG)
    hpo_params[NUM_EPOCHS] = epochs

    hpo_iter = select_integer_value(MAX_HPO_ITERS_PRINT_MSG, MAX_HPO_ITERS_PROMPT_MSG,
                                    MAX_HPO_ITERS_ERROR_MSG)
    hpo_params[NUM_OF_MAX_HPO_ITERS] = hpo_iter

    return hpo_params


def get_data_input_path(print_msg):
    print(print_msg)

    is_valid_input = False

    while is_valid_input == False:
        user_input = prompt('> Path:')

        if not os.path.exists(os.path.dirname(user_input)):
            print('Path doesn\'t exist, please type in new path')
        else:
            return user_input


def select_ratio_for_validation_set():
    print('Select the ratio of the training set used for validation (e.g. 0.5):')
    is_valid_input = False

    while is_valid_input == False:
        user_input = prompt('> Ratio: ')

        try:
            ratio = float(user_input)
            if ratio > 0. and ratio < 1.:
                return ratio
            else:
                print('Invalid input, please type in a number > 0. and < 1.')
            return ratio
        except ValueError:
            print('Invalid input, please type in a number > 0. and < 1.')

    return ratio


def is_validation_set_provided():
    print('Do you provide a validation set?')
    is_valid_input = False

    while is_valid_input == False:
        user_input = prompt('> \'yes\' or \'no\': ')

        if user_input != 'yes' and user_input != 'no':
            print('Invalid input, please type in \'yes\' or \'no\'')
        else:
            return mapping[user_input]


def select_preferred_device():
    print('Do you want to use a GPU if available?')
    is_valid_input = False

    while not is_valid_input:
        user_input = prompt('> \'yes\' or \'no\':')

        if user_input == 'yes' or user_input == 'no':
            return user_input
        else:
            print('Invalid input, please type in \'yes\' or \'no\'')


def select_integer_value(print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False

    while not is_valid_input:
        user_input = prompt(prompt_msg)

        if user_input.isnumeric():
            return int(user_input)
        else:
            print(error_msg)


def select_entites_normalization():
    print('Please select the normalization approach for the entities:')
    print('L1-Normalization: 1')
    print('L2-Normalization: 2')
    is_valid_input = False

    while not is_valid_input:
        is_valid_input = True
        user_input = prompt('> Normalization approach:')

        if user_input == '1' or user_input == '2':
            return int(user_input)
        else:
            print('Invalid input, please type in \'1\' or \'2\'')


def select_embedding_model_params(model_id):
    kg_model_params = OrderedDict()
    kg_model_params[KG_EMBEDDING_MODEL] = embedding_models_mapping[model_id]

    if 1 <= model_id and model_id <= 4:
        embedding_dimension = select_integer_value(EMBEDDING_DIMENSION_PRINT_MSG, EMBEDDING_DIMENSION_PROMPT_MSG,
                                                   EMBEDDING_DIMENSION_ERROR_MSG)

        kg_model_params[EMBEDDING_DIM] = embedding_dimension

        if model_id == 1:
            kg_model_params[NORMALIZATION_OF_ENTITIES] = select_entites_normalization()

        kg_model_params[MARGIN_LOSS] = select_float_value(MARGIN_LOSS_PRINT_MSG, MARGIN_LOSS_PROMPT_MSG,
                                                          MARGIN_LOSS_ERROR_MSG)

    kg_model_params[LEARNING_RATE] = select_float_value(LEARNING_RATE_PRINT_MSG, LEARNING_RATE_PROMPT_MSG,
                                                        LEARNING_RATE_ERROR_MSG)

    kg_model_params[BATCH_SIZE] = select_integer_value(BATCH_SIZE_PRINT_MSG, BATCH_SIZE_PROMPT_MSG,
                                                       BATCH_SIZE_ERROR_MSG)

    kg_model_params[NUM_EPOCHS] = select_integer_value(EPOCH_PRINT_MSG, EPOCH_PROMPT_MSG, EPOCH_ERROR_MSG)

    return kg_model_params


def select_float_value(print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False

    while not is_valid_input:
        user_input = prompt(prompt_msg)

        for float_value in user_input:
            try:
                float_value = float(float_value)
                return float_value
            except ValueError:
                print(error_msg)
                break


def ask_for_existing_configuration():
    print('Do you provide an existing configuration dictionary?')
    is_valid_input = False

    while is_valid_input == False:
        user_input = prompt('> \'yes\' or \'no\':')
        if user_input == 'yes' or user_input == 'no':
            return mapping[user_input]
        else:
            print('Invalid input, type \'yes\' or \'no\'')


def load_config_file():
    is_valid_input = False
    config_file_path = get_data_input_path(print_msg=CONFIG_FILE_PRINT_MSG)

    while is_valid_input == False:
        with open(config_file_path, 'rb') as f:
            try:
                data = pickle.load(f)
                assert type(data) == dict or type(data) == OrderedDict
                return data
            except:
                print('Invalid file, configuration file must be serialised dictionary (.pkl)')
                config_file_path = get_data_input_path(print_msg=CONFIG_FILE_PRINT_MSG)


def ask_binary_question(print_msg, prompt_msg, error_msg):
    print(print_msg)
    is_valid_input = False

    while is_valid_input == False:
        user_input = prompt(prompt_msg)
        if user_input == 'yes' or user_input == 'no':
            return mapping[user_input]
        else:
            print(error_msg)


def get_output_directory():
    print('Please type in the path to the output directory')
    is_valid_input = False

    while is_valid_input == False:
        user_input = prompt('> Path to output director:')
        if os.path.exists(os.path.dirname(user_input)):
            return user_input
        else:
            print('Invalid input, please type in the path to an existing directory.')


def start_cli():
    config = OrderedDict()

    print_welcome_message()
    configuration_exits = ask_for_existing_configuration()

    if configuration_exits:
        config = load_config_file()
        config[OUTPUT_DIREC] = get_output_directory()
        return config

    print('----------------------------')
    exec_mode = select_execution_mode()
    exec_mode = execution_mode_mapping[exec_mode]
    print('----------------------------')
    embedding_model_id = select_embedding_model()
    print('----------------------------')

    if exec_mode == HYPER_PARAMTER_SEARCH:
        hpo_params = select_hpo_params(model_id=embedding_model_id)
        config[HYPER_PARAMTER_OPTIMIZATION_PARAMS] = hpo_params
    else:
        kg_model_params = select_embedding_model_params(model_id=embedding_model_id)
        config[KG_EMBEDDING_MODEL] = kg_model_params

    print('----------------------------')
    eval_metrics = select_eval_metrics()
    config[EVAL_METRICS] = eval_metrics

    if HITS_AT_K in eval_metrics:
        k = select_integer_value(K_FOR_HITS_AT_K_PRINT_MSG, K_FOR_HITS_AT_K_PROMPT_MSG, K_FOR_HITS_AT_K_ERROR_MSG)
        config[K_FOR_HITS_AT_K] = k
    print('----------------------------')

    config[TRAINING_SET_PATH] = get_data_input_path(print_msg=TRAINING_SET_PRINT_MSG)

    use_validation_set = is_validation_set_provided()

    if use_validation_set:
        config[VALIDATION_SET_PATH] = get_data_input_path(print_msg=VALIDATION_SET_PRINT_MSG)
    else:
        config[VALIDATION_SET_RATIO] = select_ratio_for_validation_set()

    print('----------------------------')
    config[PREFERRED_DEVICE] = select_preferred_device()

    config[OUTPUT_DIREC] = get_output_directory()

    return config


def main():
    config = start_cli()

    pipeline = Pipeline(config=config, seed=2)

    if HYPER_PARAMTER_OPTIMIZATION_PARAMS in config:
        trained_model, eval_summary, entity_to_embedding, relation_to_embedding, params = pipeline.start_hpo()
    else:
        trained_model, eval_summary, entity_to_embedding, relation_to_embedding, params = pipeline.start_training()

    print(eval_summary)

    output_direc = config[OUTPUT_DIREC]

    out_path = os.path.join(output_direc, 'configuration.pkl')
    with open(out_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    out_path = os.path.join(output_direc, 'entities_to_embeddings.pkl')
    with open(out_path, 'wb') as handle:
        pickle.dump(entity_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

    out_path = os.path.join(output_direc, 'relations_to_embeddings.pkl')
    with open(out_path, 'wb') as handle:
        pickle.dump(relation_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
