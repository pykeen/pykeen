# -*- coding: utf-8 -*-

"""Click options for building magical KGE model CLIs."""

from typing import Optional

import click

from .. import get_model_cls
from ...evaluation import _EVALUATOR_SUFFIX, evaluators, get_evaluator_cls
from ...losses import _LOSS_SUFFIX, get_loss_cls, losses
from ...optimizers import get_optimizer_cls, optimizers
from ...stoppers import _STOPPER_SUFFIX, get_stopper_cls, stoppers
from ...training import _TRAINING_LOOP_SUFFIX, get_training_loop_cls, training_loops
from ...triples import TriplesFactory
from ...utils import normalize_string, random_non_negative_int, resolve_device


def _make_callback(f):
    def _callback(_, __, value):
        return f(value)

    return _callback


def _make_instantiation_callback(f):
    def _callback(_, __, value):
        return f(value)()

    return _callback


def _get_default(f, suffix=None):
    return normalize_string(f(None).__name__, suffix=suffix)


def triples_factory_callback(_, __, path: Optional[str]) -> Optional[TriplesFactory]:
    """Generate a triples factory using the given path."""
    return path and TriplesFactory(path=path)


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
    'loss': click.option(
        '--loss',
        type=click.Choice(losses),
        callback=_make_instantiation_callback(get_loss_cls),
        default=_get_default(get_loss_cls, suffix=_LOSS_SUFFIX),
        show_default=True,
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
    'input_dropout': click.option(
        '--input-dropout',
        type=float,
        default=0.2,
        show_default=True,
    ),
    'inner_model': click.option(
        '--inner-model',
        callback=_make_callback(get_model_cls),
        default='distmult',
        show_default=True,
    ),
    'automatic_memory_optimization': click.option(
        '--automatic-memory-optimization',
        type=bool,
        default=True,
        show_default=True,
    ),
}

device_option = click.option(
    '--device',
    callback=_make_callback(resolve_device),
    help='Can either be gpu/cuda or cuda:<ID>. Defaults to cuda, if available.',
)
optimizer_option = click.option(
    '-o', '--optimizer',
    type=click.Choice(list(optimizers)),
    default=_get_default(get_optimizer_cls),
    show_default=True,
    callback=_make_callback(get_optimizer_cls),
)
evaluator_option = click.option(
    '--evaluator',
    type=click.Choice(list(evaluators)),
    show_default=True,
    default=_get_default(get_evaluator_cls, suffix=_EVALUATOR_SUFFIX),
    callback=_make_callback(get_evaluator_cls),
)
training_loop_option = click.option(
    '--training-loop',
    type=click.Choice(list(training_loops)),
    callback=_make_callback(get_training_loop_cls),
    default=_get_default(get_training_loop_cls, suffix=_TRAINING_LOOP_SUFFIX),
    show_default=True,
)
stopper_option = click.option(
    '--stopper',
    type=click.Choice(list(stoppers)),
    callback=_make_callback(get_stopper_cls),
    default=_get_default(get_stopper_cls, suffix=_STOPPER_SUFFIX),
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
    callback=triples_factory_callback,
    help='Path to training data',
)
testing_option = click.option(
    '-q', '--testing-triples-factory',
    callback=triples_factory_callback,
    help='Path to testing data. If not supplied, then evaluation occurs on training data.',
)
valiadation_option = click.option(
    '--validation-triples-factory',
    callback=triples_factory_callback,
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
