# -*- coding: utf-8 -*-

"""Click options for building magical KGE model CLIs."""

import random

import click

from .callbacks import criterion_callback, device_callback, optimizer_callback, triples_factory_callback
from .constants import criteria_map, optimizer_map

CLI_OPTIONS = {
    'triples_factory': click.option(
        '-t', '--triples-factory',
        callback=triples_factory_callback,
        required=True,
        help='Path to training data',
    ),
    'preferred_device': click.option(
        '--preferred-device',
        callback=device_callback,
        help='Defaults to cpu. Can either be gpu/cuda or cuda:<ID>',
    ),
    'embedding_dim': click.option(
        '--embedding-dim',
        type=int,
        default=50,
        show_default=True,
    ),
    'epsilon': click.option(
        '--epsilon',
        type=float,
        default=0.005,
        show_default=True,
    ),
    'criterion': click.option(
        '--criterion',
        type=click.Choice(criteria_map),
        callback=criterion_callback,
    ),
    'random_seed': click.option(
        '--random_seed',
        type=int,
        default=random.randint(0, 2 ** 32 - 1),
    ),
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
    'input_dropout': click.option(  # ComplexCWA, ComplexLiteralCWA
        '--input-dropout',
        type=float,
        default=0.2,
        show_default=True,
    ),
}
optimizer_option = click.option(
    '--optimizer',
    type=click.Choice(list(optimizer_map)),
    default='SGD',
    show_default=True,
    callback=optimizer_callback,
)
closed_world_option = click.option(
    '--closed-world',
    is_flag=True,
)
number_epochs_option = click.option(
    '--number-epochs',
    type=int,
    default=5,
    show_default=True,
)
batch_size_option = click.option(
    '--batch-size',
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
testing_option = click.option(
    '-q', '--testing',
    callback=triples_factory_callback,
    help='Path to testing data. If not supplied, then evaluation occurs on training data.',
)
