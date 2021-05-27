# -*- coding: utf-8 -*-

"""The easiest way to optimize a model is with the :func:`pykeen.hpo.hpo_pipeline` function.

All of the following examples are about getting the best model
when training TransE on the Nations dataset. Each gives a bit
of insight into usage of the :func:`hpo_pipeline` function.

The minimal usage of the hyper-parameter optimization is to specify the
dataset, the model, and how much to run. The following example shows how to
optimize the TransE model on the Nations dataset a given number of times using
the ``n_trials`` argument.

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_pipeline_result = hpo_pipeline(
...     n_trials=30,
...     dataset='Nations',
...     model='TransE',
... )

Alternatively, the ``timeout`` can be set. In the following example,
as many trials as possible will be run in 60 seconds.

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_pipeline_result = hpo_pipeline(
...    timeout=60,
...    dataset='Nations',
...    model='TransE',
... )

Every model in PyKEEN has default values for its hyper-parameters chosen from the best-reported values in each model's
original paper unless otherwise stated on the model's reference page. In case hyper-parameters for a model for a
specific dataset were not available, we choose the hyper-parameters based on the findings in our
large-scale benchmarking [ali2020a]_.

In addition to reasonable default hyper-parameters, every model in PyKEEN has
default "strategies" for optimizing these hyper-parameters which either constitute
ranges for integer/floating point numbers or as enumerations for categorical variables
and booleans.

While the default values for hyper-parameters are encoded with the python syntax
for default values of the ``__init__()`` function of each model, the ranges/scales can be
found in the class variable :py:attr:`pykeen.models.Model.hpo_default`. For
example, the range for TransE's embedding dimension is set to optimize
between 50 and 350 at increments of 25 in :py:attr:`pykeen.models.TransE.hpo_default`.
TransE also has a scoring function norm that will be optimized by a categorical
selection of {1, 2} by default.

.. note ::

   These hyper-parameter ranges were chosen as reasonable defaults for the benchmark
   datasets FB15k-237 / WN18RR. When using different datasets, the ranges might be suboptimal.

All hyper-parameters defined in the ``hpo_default`` of your chosen model will be
optimized by default. If you already have a value that you're happy with for
one of them, you can specify it with the ``model_kwargs`` attribute. In the
following example, the ``embedding_dim`` for a TransE model is fixed at 200,
while the rest of the parameters will be optimized using the pre-defined HPO strategies in
the model. For TransE, that means that the scoring function norm will be optimized
as 1 or 2.

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_pipeline_result = hpo_pipeline(
...    model='TransE',
...    model_kwargs=dict(
...        embedding_dim=200,
...    ),
...    dataset='Nations',
...    n_trials=30,
... )

If you would like to set your own HPO strategy for the model's hyperparameters, you can do so with the
``model_kwargs_ranges`` argument. In the example below, the embeddings are
searched over a larger range (``low`` and ``high``), but with a higher step
size (``q``), such that 100, 200, 300, 400, and 500 are searched.

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_result = hpo_pipeline(
...     n_trials=30,
...     dataset='Nations',
...     model='TransE',
...     model_kwargs_ranges=dict(
...         embedding_dim=dict(type=int, low=100, high=500, q=100),
...     ),
... )

.. warning::

    If the given range is not divisible by the step size, then the
    upper bound will be omitted.

If you want to optimize the entity initializer, you can use the `categorical` type,
which requres a `choices` key with a list of choices. This works for strings, integers,
floats, etc.

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_result = hpo_pipeline(
...     n_trials=30,
...     dataset='Nations',
...     model='TransE',
...     model_kwargs_ranges=dict(
...         entity_initializer=dict(type='categorical', choices=[
...             'xavier_uniform',
...             'xavier_uniform_norm',
...             'uniform',
...         ]),
...     ),
... )

The same could be used for constrainers, normalizers, and regularizers over both entities and
relations. However, different models might have different names for the initializer, normalizer,
constrainer and regularizer since there could be multiple representations for either the entity,
relation, or both. Check your desired model's documentation page for the kwargs that you can
optimize over.

The HPO pipeline does not support optimizing over the hyper-parameters for each
initializer. If you are interested in this, consider rolling your own ablation
study pipeline.

Optimizing Different Components
-------------------------------

Optimizing the Loss
~~~~~~~~~~~~~~~~~~~
While each model has its own default loss, you can explicitly specify a loss
the same way as in :func:`pykeen.pipeline.pipeline`.

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_pipeline_result = hpo_pipeline(
...    n_trials=30,
...    dataset='Nations',
...    model='TransE',
...    loss='MarginRankingLoss',
... )

As stated in the documentation for :func:`pykeen.pipeline.pipeline`, each model
specifies its own default loss function in :py:attr:`pykeen.models.Model.loss_default`.
For example, the TransE model defines the margin ranking loss as its default in
:py:attr:`pykeen.models.TransE.loss_default`.

Each model also specifies default hyper-parameters for the loss function in
:py:attr:`pykeen.models.Model.loss_default_kwargs`. For example, DistMultLiteral
explicitly sets the margin to `0.0` in  :py:attr:`pykeen.models.DistMultLiteral.loss_default_kwargs`.

Unlike the model's hyper-parameters, the models don't store the strategies for
optimizing the loss functions' hyper-parameters. The pre-configured strategies
are stored in the loss function's class variable :py:attr:`pykeen.models.Loss.hpo_default`.

However, similarily to how you would specify ``model_kwargs_ranges``, you can
specify the ``loss_kwargs_ranges`` explicitly, as in the following example.

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_pipeline_result = hpo_pipeline(
...    n_trials=30,
...    dataset='Nations',
...    model='TransE',
...    loss='MarginRankingLoss',
...    loss_kwargs_ranges=dict(
...        margin=dict(type=float, low=1.0, high=2.0),
...    ),
... )



Optimizing the Negative Sampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When the stochastic local closed world assumption (sLCWA) training approach is used for training, a negative sampler
(subclass of :py:class:`pykeen.sampling.NegativeSampler`) is chosen.
Each has a strategy stored in :py:attr:`pykeen.sampling.NegativeSampler.hpo_default`.

Like models and regularizers, the rules are the same for specifying ``negative_sampler``,
``negative_sampler_kwargs``, and ``negative_sampler_kwargs_ranges``.

Optimizing the Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~
Yo dawg, I heard you liked optimization, so we put an optimizer around your
optimizer so you can optimize while you optimize. Since all optimizers used
in PyKEEN come from the PyTorch implementations, they obviously do not have
``hpo_defaults`` class variables. Instead, every optimizer has a default
optimization strategy stored in :py:attr:`pykeen.optimizers.optimizers_hpo_defaults`
the same way that the default strategies for losses are stored externally.

Optimizing Everything Else
~~~~~~~~~~~~~~~~~~~~~~~~~~
Without loss of generality, the following arguments to :func:`pykeen.pipeline.pipeline`
have corresponding `*_kwargs` and `*_kwargs_ranges`:

- ``training_loop`` (only kwargs, not kwargs_ranges)
- ``evaluator``
- ``evaluation``

Early Stopping
--------------
Early stopping can be baked directly into the :mod:`optuna` optimization.

The important keys are ``stopper='early'`` and ``stopper_kwargs``.
When using early stopping, the :func:`hpo_pipeline` automatically takes
care of adding appropriate callbacks to interface with :mod:`optuna`.

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_pipeline_result = hpo_pipeline(
...     n_trials=30,
...     dataset='Nations',
...     model='TransE',
...     stopper='early',
...     stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
... )

These stopper kwargs were chosen to make the example run faster. You will
likely want to use different ones.

Configuring Optuna
------------------
Choosing a Search Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Because PyKEEN's hyper-parameter optimization pipeline is powered by Optuna, it can directly use all of
Optuna's built-in samplers listed on :mod:`optuna.samplers` or any custom subclass of
:class:`optuna.samplers.BaseSampler`.

By default, PyKEEN uses the Tree-structured Parzen Estimator (TPE; :class:`optuna.samplers.TPESampler`),
a probabilistic search algorithm. You can explicitly set the sampler using the ``sampler`` argument
(not to be confused with the negative sampler used when training under the sLCWA):

>>> from pykeen.hpo import hpo_pipeline
>>> from optuna.samplers import TPESampler
>>> hpo_pipeline_result = hpo_pipeline(
...    n_trials=30,
...    sampler=TPESampler,
...    dataset='Nations',
...    model='TransE',
... )

You can alternatively pass a string so you don't have to worry about importing Optuna. PyKEEN knows that sampler
classes always end in "Sampler" so you can pass either "TPE" or "TPESampler" as a string. This is case-insensitive.

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_pipeline_result = hpo_pipeline(
...    n_trials=30,
...    sampler="tpe",
...    dataset='Nations',
...    model='TransE',
... )

It's also possible to pass a sampler instance directly:

>>> from pykeen.hpo import hpo_pipeline
>>> from optuna.samplers import TPESampler
>>> sampler = TPESampler(prior_weight=1.1)
>>> hpo_pipeline_result = hpo_pipeline(
...    n_trials=30,
...    sampler=sampler,
...    dataset='Nations',
...    model='TransE',
... )

If you're working in a JSON-based configuration setting, you won't be able to instantiate the sampler
with your desired settings like this. As a solution, you can pass the keyword arguments via the
``sampler_kwargs`` argument in combination with specifying the sampler as a string/class to the HPO pipeline like in:

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_pipeline_result = hpo_pipeline(
...    n_trials=30,
...    sampler="tpe",
...    sampler_kwargs=dict(prior_weight=1.1),
...    dataset='Nations',
...    model='TransE',
... )

To emulate most hyper-parameter optimizations that have used random
sampling, use :class:`optuna.samplers.RandomSampler` like in:

>>> from pykeen.hpo import hpo_pipeline
>>> from optuna.samplers import RandomSampler
>>> hpo_pipeline_result = hpo_pipeline(
...    n_trials=30,
...    sampler=RandomSampler,
...    dataset='Nations',
...    model='TransE',
... )

Grid search can be performed using :class:`optuna.samplers.GridSampler` like in:

>>> from pykeen.hpo import hpo_pipeline
>>> from optuna.samplers import GridSampler
>>> hpo_pipeline_result = hpo_pipeline(
...    n_trials=30,
...    sampler=GridSampler,
...    dataset='Nations',
...    model='TransE',
... )

Full Examples
-------------
The examples above have shown the permutation of one setting at a time. This
section has some more complete examples.

The following example sets the optimizer, loss, training, negative sampling,
evaluation, and early stopping settings.

>>> from pykeen.hpo import hpo_pipeline
>>> hpo_pipeline_result = hpo_pipeline(
...     n_trials=30,
...     dataset='Nations',
...     model='TransE',
...     model_kwargs=dict(embedding_dim=20, scoring_fct_norm=1),
...     optimizer='SGD',
...     optimizer_kwargs=dict(lr=0.01),
...     loss='marginranking',
...     loss_kwargs=dict(margin=1),
...     training_loop='slcwa',
...     training_kwargs=dict(num_epochs=100, batch_size=128),
...     negative_sampler='basic',
...     negative_sampler_kwargs=dict(num_negs_per_pos=1),
...     evaluator_kwargs=dict(filtered=True),
...     evaluation_kwargs=dict(batch_size=128),
...     stopper='early',
...     stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
... )

If you have the configuration as a dictionary:

>>> from pykeen.hpo import hpo_pipeline_from_config
>>> config = {
...     'optuna': dict(
...         n_trials=30,
...     ),
...     'pipeline': dict(
...         dataset='Nations',
...         model='TransE',
...         model_kwargs=dict(embedding_dim=20, scoring_fct_norm=1),
...         optimizer='SGD',
...         optimizer_kwargs=dict(lr=0.01),
...         loss='marginranking',
...         loss_kwargs=dict(margin=1),
...         training_loop='slcwa',
...         training_kwargs=dict(num_epochs=100, batch_size=128),
...         negative_sampler='basic',
...         negative_sampler_kwargs=dict(num_negs_per_pos=1),
...         evaluator_kwargs=dict(filtered=True),
...         evaluation_kwargs=dict(batch_size=128),
...         stopper='early',
...         stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
...     )
... }
... hpo_pipeline_result = hpo_pipeline_from_config(config)

If you have a configuration (in the same format) in a JSON file:

>>> import json
>>> config = {
...     'optuna': dict(
...         n_trials=30,
...     ),
...     'pipeline': dict(
...         dataset='Nations',
...         model='TransE',
...         model_kwargs=dict(embedding_dim=20, scoring_fct_norm=1),
...         optimizer='SGD',
...         optimizer_kwargs=dict(lr=0.01),
...         loss='marginranking',
...         loss_kwargs=dict(margin=1),
...         training_loop='slcwa',
...         training_kwargs=dict(num_epochs=100, batch_size=128),
...         negative_sampler='basic',
...         negative_sampler_kwargs=dict(num_negs_per_pos=1),
...         evaluator_kwargs=dict(filtered=True),
...         evaluation_kwargs=dict(batch_size=128),
...         stopper='early',
...         stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
...     )
... }
... with open('config.json', 'w') as file:
...    json.dump(config, file, indent=2)
... hpo_pipeline_result = hpo_pipeline_from_path('config.json')

.. seealso::

   - https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f  # noqa:E501
"""

from .hpo import HpoPipelineResult, hpo_pipeline, hpo_pipeline_from_config, hpo_pipeline_from_path  # noqa: F401

__all__ = [
    'HpoPipelineResult',
    'hpo_pipeline_from_path',
    'hpo_pipeline_from_config',
    'hpo_pipeline',
]
