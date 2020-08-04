# -*- coding: utf-8 -*-

"""Ablation studies in PyKEEN.

  "If :func:`pykeen.pipeline.pipeline` were making a maze, :func:`pykeen.hpo.hpo_pipeline` were getting
  killed by your dead wife, then :func:`pykeen.ablation.ablation_pipeline` is washing up on
  a beach." -Christopher Nolan

An ablation study is an experiment in which components of a machine learning system are removed/replaced in order
to measure the impact of these components on the system's performance.
In PyKEEN, a user could for instance measure the impact of explicitly modeling inverse relations on the model's
performance.

An ablation study is defined based on a dictionary containing three sub-dictionaries: 1.) metadata,
2.) ablation, and 3.) optuna, which we will define below.
In the following, we have to define in several cases the dictionaries 'kwargs' and 'kwargs_ranges.
'kwargs' defines for a specific component (e.g., interaction model or loss function) single hyper-parameter values
such as a fixed embedding dimension of 50, whereas 'kwargs_ranges' define ranges of values.
Note that we always have to define both dictionaries, and in cases where do not have entries for
'kwargs' or 'kwargs_ranges', we define empty dictionaries.

>>> configuration = {}

----------------Define Metadata Dictionary----------------
>>> metadata = dict(
...    title="HPO over MyData"
... )

>>> configuration['metadata'] = metadata

----------------Define Ablation Dictionary----------------
>>> ablation = {}

# Step 1: define dataset. Here, we use our own data.
>>> datasets = [
...    dict(
...        training='/path/to/my/train.txt',
...        testing='/path/to/my/test.txt',
...        validation='/path/to/my/valid.txt',
...    )
... ]

>>> ablation['datasets'] = datasets

Step 2: define model (several models can be defined).
Note the structure of 'model_kwargs': model_kwargs{InteractionModel:{parameter={range}}}.
>>> models = ['RotatE']
>>> model_kwargs = dict(
...    RotatE=dict(automatic_memory_optimization=True)
... )
>>> model_kwargs_ranges = dict(
...    RotatE=dict(
...        embedding_dim=dict(
...            type='int',
...            low=3,
...            high=5,
...            scale='power_two',
...        )
...     )
... )

Define, whether interaction model should explicitly be trained with inverse relations. If set to 'True',
the number of relations are doubled, and the task of predicting the head entities of (r,t)-pairs, becomes the task
of predicting tail entities of (t,r_inv)-pairs.
>>> create_inverse_triples = [True, False]

Define regularize (several regularizers can be defined). Here we use 'NoRegularizer' to indicate that
we do not regularize our model.
>>> regularizers = ['NoRegularizer']
>>> regularizer_kwargs = dict(RotatE=dict(NoRegularizer=dict()))
>>> regularizer_kwargs_ranges = dict(RotatE=dict(NoRegularizer=dict()))

>>> ablation['models'] = models
>>> ablation['model_kwargs'] = model_kwargs
>>> ablation['model_kwargs_ranges'] = model_kwargs_ranges
>>> ablation['create_inverse_triples'] = create_inverse_triples
>>> ablation['regularizers'] = regularizers
>>> ablation['regularizer_kwargs'] = regularizer_kwargs
>>> ablation['regularizer_kwargs_ranges'] = regularizer_kwargs_ranges

Step 3: define loss function (several loss functions can be defined). Here focus on the negative sampling
self adversarial loss.
>>> loss_functions = ['NSSALoss']
Define that we do not have 'loss_kwargs' for NSSALoss.
>>> loss_kwargs = dict(RotatE=dict(NSSALoss=dict()))
>>> loss_kwargs_ranges = dict(
...    RotatE=dict(
...        NSSALoss=dict(
...            margin=dict(
...                type='float',
...                low=1,
...                high=30,
...                q=2.0,
...            ),
...            adversarial_temperature=dict(
...                type='float',
...                low=0.1,
...                high=1.0,
...                q=0.1,
...            )
...        )
...    )
... )

>>> ablation['loss_functions'] = loss_functions
>>> ablation['loss_kwargs'] = loss_kwargs
>>> ablation['loss_kwargs_ranges'] = loss_kwargs_ranges

Step 4: define training approach: sLCWA and/or LCWA
>>> training_loops = ['sLCWA']
>>> ablation['training_loops'] = training_loops

Define negative sampler. Since we are using the sLCWA training approach, we define a negative sampler.
>>> negative_sampler = 'BasicNegativeSampler'
Define that we do not have 'negative_sampler_kwargs'.
>>> negative_sampler_kwargs = dict(RotatE=dict(BasicNegativeSampler=dict()))
>>> negative_sampler_kwargs_ranges = dict(
...    RotatE=dict(
...        BasicNegativeSampler=dict(
...            num_negs_per_pos=dict(
...                type='int',
...                low=1,
...                high=10,
...                q=1,
...            )
...        )
...    )
... )

>>> ablation['negative_sampler'] = negative_sampler
>>> ablation['negative_sampler_kwargs'] = negative_sampler_kwargs
>>> ablation['negative_sampler_kwargs_ranges'] = negative_sampler_kwargs_ranges

Step 5: define optimizer (several optimizers can be defined).
>>> optimizers = ['adam']
>>> optimizer_kwargs = dict(
...    RotatE=dict(
...        adam=dict(
...            weight_decay=0.0
...        )
...    )
... )
>>> optimizer_kwargs_ranges = dict(
...    RotatE=dict(
...        adam=dict(
...            lr=dict(
...                type='float',
...                low=0.001,
...                high=0.1,
...                sclae='log',
...            )
...        )
...    )
... )

>>> ablation['optimizers'] = optimizers
>>> ablation['optimizer_kwargs'] = optimizer_kwargs
>>> ablation['optimizer_kwargs_ranges'] = optimizer_kwargs_ranges

Step 6: define training parameters.
>>> training_kwargs = dict(
...    RotatE=dict(
...        sLCWA=dict(
...            num_epochs=10,
...            label_smoothing=0.0,
...        )
...    )
... )
>>> training_kwargs_ranges = dict(
...    RotatE=dict(
...        sLCWA=dict(
...            batch_size=dict(
...                type='int',
...                low=6,
...                high=8,
...                scale='power_two',
...            )
...        )
...    )
... )

>>> ablation['training_kwargs'] = training_kwargs
>>> ablation['training_kwargs_ranges'] = training_kwargs_ranges

Step 7: define evaluator.
>>> evaluator = 'RankBasedEvaluator'
>>> evaluator_kwargs = dict(
...    filtered=True,
)
>>> evaluation_kwargs = dict(
...    batch_size=None  # searches for maximal possible in order to minimize evaluation time
)

>>> ablation['evaluator'] = evaluator
>>> ablation['evaluator_kwargs'] = evaluator_kwargs
>>> ablation['evaluation_kwargs'] = evaluation_kwargs

Step 8: define early stopper.
>>> stopper = 'early'
>>> stopper_kwargs = dict(
...    frequency=50,
...    patience=2,
...    delta=0.002,
... )

>>> ablation['stopper'] = stopper
>>> ablation['stopper_kwargs'] = stopper_kwargs

>>> configuration['ablation'] = ablation

----------------Define Optuna Dictionary----------------

Define Optuna related parameters.
>>> optuna = {}
Define the number of HPO iterations, where in each iteration new hyper-parameters will be sampled.
>>> optuna['n_trials'] = 2
Define the ablation study's timeout. An ablation study will be terminated after the timeout is reached
independently of the defined number of 'n_trials.' Note: every HPO iteration that has been started before
the timeout has been reached, will be finished before terminating the current ablation study.
>>> optuna['timeout'] = 10
Define the metric to optimize.
>>> optuna['metric'] = 'hits@10'
Define whether to 'maximize' or 'minimize' the metric.
>>> optuna['direction'] = 'maximize'
Define the HPO sampler. Here, we use random search.
>>> optuna['sampler'] = 'random'
Define which Optuna pruner should be used. We indicate that we do not use an pruner.
>>> optuna['pruner'] = 'nop'

>>> configuration['optuna'] = optuna

>>> output_directory = '/Users/mali/PycharmProjects/pykeen_1_0/data'
Defines how often the model should be re-trained and evaluated based on the best hyper-parameters which
enables us to measure the variance in performance.
>>> best_replicates = 2
Defines, whether each trained model sampled during HPO should be saved.
>>> save_artifacts = False
Defines, whether the best model should be discarded after training and evaluation.
>>> discard_replicates = False
Defines, whether a replicate of the best model should be moved to CPU.
We recommend to set this flag to 'True' to avoid unnecessary GPU usage.
>>> move_to_cpu = True

>>> ablation_pipeline(
...    config=configuration,
...    directory=output_directory,
...    best_replicates=best_replicates,
...    save_artifacts=save_artifacts,
...    discard_replicates=discard_replicates,
...    move_to_cpu=move_to_cpu,
...    dry_run=False,
... )

.. warning::

  do not use JSON files for examples. All usage should be through the python dictionary interface. If we
  forgot to make a python dictionary interface, then we must add one for this PR

"""

from .ablation import ablation_pipeline, prepare_ablation, prepare_ablation_from_config

__all__ = [
    'ablation_pipeline',
    'prepare_ablation_from_config',
    'prepare_ablation',
]
