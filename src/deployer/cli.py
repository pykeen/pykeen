import os
import sys
from collections import OrderedDict

from prompt_toolkit import prompt

from utilities.constants import PREFERRED_DEVICE, GPU, CPU

w_dir = os.path.dirname(os.getcwd())
sys.path.append(w_dir)

from utilities.pipeline import Pipeline

mapping = {'yes': True, 'no': False}
embedding_models_mapping = {1: 'TransE', 2: 'TransH', 3: 'TransR', 4: 'TransD'}
metrics_maping = {1: 'mean_rank'}


def start_cli():
    config = OrderedDict()

    print('Do you want to apply a hyper-parameter search?')
    user_input = prompt('> Please type \'yes\' or \'no\': ')

    assert user_input == 'yes' or user_input == 'no'

    apply_hpo = mapping[user_input]

    print('----------------------------')

    embedding_model_id = select_embedding_model()

    print('----------------------------')

    if apply_hpo:
        hpo_params = select_hpo_params(model_id=embedding_model_id)
        config['hyper_param_optimization'] = hpo_params
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


def select_hpo_params(model_id):
    hpo_params = OrderedDict()
    hpo_params['kg_embedding_model'] = embedding_models_mapping[model_id]

    if 1 <= model_id and model_id <= 4:
        print('Please type the range of preferred embedding dimensions comma separated (e.g. 50,100,200):')
        user_input = prompt('> Embedding dimensions: ')
        embedding_dimensions = user_input.split(',')
        embedding_dimensions = [int(emb) for emb in embedding_dimensions]
        hpo_params['embedding_dim'] = embedding_dimensions

        print('Please type the range of preferred maring losses comma separated  (e.g. 1,2,10):')
        user_input = prompt('> Margin losses: ')
        margin_losses = user_input.split(',')
        margin_losses = [float(margin_loss) for margin_loss in margin_losses]
        hpo_params['margin_loss'] = margin_losses
    else:
        # TODO: Change
        exit(0)

    # General parmas
    print('Please type the range of preferred learning rates omma separated  (e.g. 0.1, 0.01, 0.0001):')
    user_input = prompt('> Learning rates: ')
    lrs = user_input.split(',')
    lrs = [float(lr) for lr in lrs]
    hpo_params['learning_rate'] = lrs

    print('Please type the range of preferred batch sizes comma separated (e.g. 32, 64, 128):')
    user_input = prompt('> Batch sizes: ')
    batch_sizes = user_input.split(',')
    batch_sizes = [int(batch_size) for batch_size in batch_sizes]
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


def select_embedding_model():
    print('Please select the embedding model you want to use:')
    print("TransE: 1")
    print("TransH: 2")
    print("TransR: 3")
    print("TransD: 4")
    user_input = int(prompt('> Please select one of the options: '))

    assert 1 <= user_input and user_input <= 4

    return user_input


def main():
    config = start_cli()

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
