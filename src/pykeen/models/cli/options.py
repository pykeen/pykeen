# -*- coding: utf-8 -*-

"""Click options for building magical KGE model CLIs."""

import click

from .. import model_resolver
from ...evaluation import evaluator_resolver
from ...losses import loss_resolver
from ...optimizers import optimizer_resolver
from ...stoppers import stopper_resolver
from ...training import training_loop_resolver
from ...utils import random_non_negative_int, resolve_device


def _make_callback(f):
    def _callback(_, __, value):
        return f(value)

    return _callback


def _make_instantiation_callback(f):
    def _callback(_, __, value):
        return f(value)()

    return _callback


CLI_OPTIONS = {
    'embedding_dim': click.option(
        '--embedding-dim',
        type=int,
        default=50,
        show_default=True,
        help='Embedding dimensions for entities.',
    ),
    'epsilon': click.option(
        '--epsilon',
        type=float,
        default=0.005,
        show_default=True,
    ),
    'loss': loss_resolver.get_option('--loss'),
    'regularization_factor': click.option(  # ComplEx
        '--regularization-factor',
        type=float,
        default=0.1,
        show_default=True,
    ),
    'scoring_fct_norm': click.option(  # SE, TransD, TransE, TransH, TransR, UM
        '--scoring-fct-norm',
        type=float,
        default=2,
        show_default=True,
        help='The p-norm to be used',
    ),
    'soft_weight_constraint': click.option(
        '--soft-weight-constraint',
        type=float,
        default=0.05,
        show_default=True,
    ),
    'relation_dim': click.option(  # TransD, TransR
        '--relation-dim',
        type=int,
        default=50,
        show_default=True,
    ),
    'neg_label': click.option(  # ComplEx
        '--neg-label',
        type=int,
        default=-1,
        show_default=True,
    ),
    'input_dropout': click.option(
        '--input-dropout',
        type=float,
        default=0.2,
        show_default=True,
    ),
    'inner_model': model_resolver.get_option('--inner-model', default='distmult'),
    'clamp_score': click.option(
        '--clamp-score',
        type=float,
    ),
    'combination_dropout': click.option(
        '--combination-dropout',
        type=float,
    ),
}

device_option = click.option(
    '--device',
    callback=_make_callback(resolve_device),
    help='Can either be gpu/cuda or cuda:<ID>. Defaults to cuda, if available.',
)
optimizer_option = optimizer_resolver.get_option('-o', '--optimizer')
evaluator_option = evaluator_resolver.get_option('--evaluator')
training_loop_option = training_loop_resolver.get_option('--training-loop')
stopper_option = stopper_resolver.get_option('--stopper')

automatic_memory_optimization_option = click.option(
    '--automatic-memory-optimization/--no-automatic-memory-optimization',
    default=True,
    show_default=True,
)

number_epochs_option = click.option(
    '-n', '--number-epochs',
    type=int,
    default=5,
    show_default=True,
)
batch_size_option = click.option(
    '-b', '--batch-size',
    type=int,
    default=256,
    show_default=True,
)
learning_rate_option = click.option(
    '--learning-rate',
    type=float,
    default=0.001,
    show_default=True,
)
dataset_option = click.option('--dataset', help='Dataset name')
training_option = click.option(
    '-t', '--training-triples-factory',
    help='Path to training data',
)
testing_option = click.option(
    '-q', '--testing-triples-factory',
    help='Path to testing data. If not supplied, then evaluation occurs on training data.',
)
valiadation_option = click.option(
    '--validation-triples-factory',
    help='Path to validation data. Must be supplied for early stopping',
)
mlflow_uri_option = click.option(
    '--mlflow-tracking-uri',
    help='MLFlow tracking URI',
)
title_option = click.option(
    '--title',
    help='Title of this experiment',
)
num_workers_option = click.option(
    '--num-workers',
    type=int,
    help='The number of child CPU worker processes used for preparing batches during training. If not specified,'
         ' batches are prepared in the main process.',
)
random_seed_option = click.option(
    '--random-seed',
    type=int,
    default=random_non_negative_int(),
    show_default=True,
    help='Random seed for PyTorch, NumPy, and Python.',
)
