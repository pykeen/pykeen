import os
import sys

w_dir = os.path.dirname(os.getcwd())
sys.path.append(w_dir)
from collections import OrderedDict

from prompt_toolkit import prompt

from utilities.constants import PREFERRED_DEVICE, GPU, CPU
from utilities.pipeline import Pipeline

# ----------Constants--------------
TRAINING = 'training'
HYPER_PARAMTER_SEARCH = 'hyper_parameter_search'
# TODO: Adapt
HYPER_PARAMTER_OPTIMIZATION_PARAMS = 'hyper_param_optimization'
EMBEDDING_DIMENSION = 'embedding_dim'
# ---------------------------------

mapping = {'yes': True, 'no': False}
embedding_models_mapping = {1: 'TransE', 2: 'TransH', 3: 'TransR', 4: 'TransD'}
metrics_maping = {1: 'mean_rank'}
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
            is_valid_input = True
            user_input = int(user_input)

    return user_input


def select_embedding_model():
    print('Please select the embedding model you want to use:')
    print("TransE: 1")
    print("TransH: 2")
    print("TransR: 3")
    print("TransD: 4")
    is_valid_input = False

    while is_valid_input == False:
        user_input = int(prompt('> Please select one of the options: '))

        if user_input != '1' and user_input != '2':
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
        user_input = prompt(prompt_msg)
        user_input = user_input.split(',')

        for integer in user_input:
            if integer.isnumeric():
                integers.append(int(integer))
            else:
                print(error_msg)
                break

        is_valid_input = True

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


def select_learning_rates():
    print('Please type the range of preferred learning rate(s) comma separated (e.g. 0.1, 0.01, 0.0001:')
    is_valid_input = False
    learning_rates = []

    while is_valid_input == False:
        user_input = prompt('> Please select the learning rate(s):')
        user_input = user_input.split(',')

        for learning_rate in user_input:
            try:
                learning_rate = float(learning_rate)
                learning_rates.append(int(learning_rate))
            except ValueError:
                print("Invalid input, please positice integer as embedding dimensions")
                break

        is_valid_input = True

    return learning_rates


def _select_trans_x_params():
    hpo_params = OrderedDict()
    print_msg = 'Please type the range of preferred embedding dimensions comma separated (e.g. 50,100,200):'
    prompt_msg = '> Please select the embedding dimensions:'
    error_msg = 'Invalid input, please positice integer as embedding dimensions.'
    embedding_dimensions = select_positive_integer_values(print_msg, prompt_msg, error_msg)
    hpo_params['embedding_dim'] = embedding_dimensions

    # ---------
    print_msg = 'Please type the range of preferred margin losse(s) comma separated  (e.g. 1,2,10):'
    prompt_msg = '> Please select the margin losse(s):'
    error_msg = 'Invalid input, please positice integer as embedding dimensions.'
    margin_losses = select_float_values(print_msg, prompt_msg, error_msg)
    hpo_params['margin_loss'] = margin_losses

    return hpo_params


def select_hpo_params(model_id):
    hpo_params = OrderedDict()
    hpo_params['kg_embedding_model'] = embedding_models_mapping[model_id]

    if 1 <= model_id and model_id <= 4:
        # Model is one of the TransX versions
        param_dict = _select_trans_x_params()
        hpo_params.update(param_dict)
    else:
        # TODO: Change
        exit(0)

    # General params
    # --------
    print_msg = 'Please type the range of preferred learning rate(s) comma separated (e.g. 0.1, 0.01, 0.0001:'
    prompt_msg = '> Please select the learning rate(s):'
    error_msg = 'Invalid input, please float values for the learning rate(s).'
    learning_rates = select_float_values(print_msg, prompt_msg, error_msg)
    hpo_params['learning_rate'] = learning_rates

    # --------------
    print_msg = 'Please type the range of preferred batch sizes comma separated (e.g. 32, 64, 128):'
    prompt_msg = '> Please select the embedding dimensions:'
    error_msg = 'Invalid input, please select integers as batch size(s)'
    batch_sizes = select_positive_integer_values(prompt_msg, prompt_msg, error_msg)
    hpo_params['batch_size'] = batch_sizes

    print('Please type the range of preferred epochs comma separated (e.g. 1, 5, 100):')
    user_input = prompt('> Epochs: ')
    epochs = user_input.split(',')
    epochs = [int(epoch) for epoch in epochs]
    hpo_params['num_epochs'] = epochs

    print('Please type the number of hyper-parameter iterations (single number):')
    user_input = prompt('> HPO iterations: ')
    hpo_iter = int(user_input)
    hpo_params['max_iters'] = hpo_iter

    return hpo_params


def start_cli():
    config = OrderedDict()

    print_welcome_message()
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
        config['kg_embedding_model'] = kg_model_params

    print('----------------------------')
    eval_metrics = select_eval_metrics()
    config['eval_metrics'] = eval_metrics

    print('----------------------------')
    print('Please provide the path to the training set')
    training_data_path = prompt('> Path: ')

    config['training_set_path'] = training_data_path

    print('Do you provide a validation set?')
    user_input = prompt('> \'yes\' or \'no\': ')
    user_input = mapping[user_input]

    if user_input:
        print('Please provide the path to the validation set')
        validation_data_path = prompt('> Path: ')
        config['validation_set_path'] = validation_data_path
    else:
        print('Select the ratio of the training set used for validation (e.g. 0.5)')
        user_input = prompt('> Ratio: ')
        validation_ratio = float(user_input)
        config['validation_set_ratio'] = validation_ratio

    print('Do you want to use a GPU if available?')
    user_input = prompt('> \'yes\' or \'no\': ')
    if user_input == 'yes':
        config[PREFERRED_DEVICE] = GPU
    else:
        config[PREFERRED_DEVICE] = CPU

    return config


def select_embedding_model_params(model_id):
    kg_model_params = OrderedDict()
    kg_model_params['model_name'] = embedding_models_mapping[model_id]

    if 1 <= model_id and model_id <= 4:
        print('Please type the embedding dimensions:')
        user_input = prompt('> Embedding dimension: ')
        embedding_dimension = int(user_input)

        kg_model_params['embedding_dim'] = embedding_dimension

        if model_id == 1:
            print('Please select the normalization approach for the entities: ')
            print('L1-Norm: 1')
            print('L1-Norm: 2')
            user_input = prompt('> Normalization approach: ')
            normalization_of_entities = int(user_input)

            kg_model_params['normalization_of_entities'] = normalization_of_entities

        print('Please type the maring loss: ')
        user_input = prompt('> Margin loss: ')
        margin_loss = int(user_input)

        kg_model_params['margin_loss'] = margin_loss

    print('Please type the learning rate: ')
    user_input = prompt('> Learning rate: ')
    lr = float(user_input)
    kg_model_params['learning_rate'] = lr

    print('Please type the batch size: ')
    user_input = prompt('> Batch size: ')
    batch_size = int(user_input)
    kg_model_params['batch_size'] = batch_size

    print('Please type the number of epochs: ')
    user_input = prompt('> Epochs: ')
    epochs = int(user_input)

    kg_model_params['num_epochs'] = epochs

    return kg_model_params


def select_eval_metrics():
    print('Please select the evaluation metrics you want to use:')
    print("Mean rank: 1")
    user_input = prompt('> Please select the options comma separated: ')
    metrics = user_input.split(',')
    metrics = [metrics_maping[int(metric)] for metric in metrics]

    return metrics


import pickle


def main():
    config = start_cli()

    config_in = '/Users/mehdi/PycharmProjects/kg_embeddings_pipeline/data/config_files/hpo_wn_18_test_test.pkl'
    with open(config_in, 'rb') as handle:
        config = pickle.load(handle)
    pipeline = Pipeline(config=config, seed=2)

    if 'hyper_param_optimization' in config:
        trained_model, eval_summary, entity_to_embedding, relation_to_embedding, params = pipeline.start_hpo()
    else:
        trained_model, eval_summary, entity_to_embedding, relation_to_embedding, params = pipeline.start_training()

    summary = eval_summary.copy()
    summary.update(params)

    print(summary)


if __name__ == '__main__':
    main()
