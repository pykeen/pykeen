"""Running an ablation study."""

# %%
# Let's start with defining the minimal requirements, i.e., the dataset(s), interaction model(s), the loss
# function(s), training approach(es), and the optimizer(s) in order to run the ablation study.
from pykeen.ablation import ablation_pipeline

directory = "doctests/ablation/ex01_minimal"
ablation_pipeline(
    directory=directory,
    models=["ComplEx"],
    datasets=["Nations"],
    losses=["BCEAfterSigmoidLoss", "MarginRankingLoss"],
    training_loops=["LCWA"],
    optimizers=["Adam"],
    # The following are not part of minimal configuration, but are necessary
    # for demonstration/doctests. You should make these numbers bigger when
    # you're using PyKEEN's ablation framework
    epochs=1,
    n_trials=1,
)

# %%
# We can provide arbitrary additional information about our study with the ``metadata`` keyword. Some keys, such
# as ``title`` are special and used by PyKEEN and :mod:`optuna`.
directory = "doctests/ablation/ex02_metadata"
ablation_pipeline(
    directory=directory,
    models=["ComplEx"],
    datasets=["Nations"],
    losses=["BCEAfterSigmoidLoss", "MarginRankingLoss"],
    training_loops=["LCWA"],
    optimizers=["Adam"],
    # Add metadata with:
    metadata={
        "title": "Ablation Study Over Nations for ComplEx.",
    },
    # Fast testing configuration, make bigger in prod
    epochs=1,
    n_trials=1,
)

# %%
# As mentioned above, we also want to measure the effect of explicitly modeling inverse relations on the model's
# performance. Therefore, we extend the ablation study by including the ``create_inverse_triples`` argument. Unlike
# ``models``, ``datasets``, ``losses``, ``training_loops``, and ``optimizers``, ``create_inverse_triples`` has a
# default value, which is ``False``.
directory = "doctests/ablation/ex03_inverse"
ablation_pipeline(
    directory=directory,
    models=["ComplEx"],
    datasets=["Nations"],
    losses=["BCEAfterSigmoidLoss"],
    training_loops=["LCWA"],
    optimizers=["Adam"],
    # Add inverse triples with
    create_inverse_triples=[True, False],
    # Fast testing configuration, make bigger in prod
    epochs=1,
    n_trials=1,
)

# %%
# If there is only one value for either the ``models``, ``datasets``, ``losses``, ``training_loops``,
# ``optimizers``, or ``create_inverse_triples`` argument, it can be given as a single value instead of the list.
# It doesn't make sense to run an ablation study if all of these values are fixed.
directory = "doctests/ablation/ex04_terse_kwargs"
ablation_pipeline(
    directory=directory,
    models="ComplEx",
    datasets="Nations",
    losses=["BCEAfterSigmoidLoss", "MarginRankingLoss"],
    training_loops="LCWA",
    optimizers="Adam",
    create_inverse_triples=[True, False],
    # Fast testing configuration, make bigger in prod
    epochs=1,
    n_trials=1,
)

# %%
# For each of the components of a knowledge graph embedding model (KGEM) that requires hyper-parameters, i.e.,
# interaction model, loss function, and the training approach, we provide default hyper-parameter optimization
# (HPO) ranges within PyKEEN. To finalize the ablation study, we recommend defining early stopping for your
# ablation study. We define the early stopper using the argument ``stopper``, and through ``stopper_kwargs``, we
# provide instantiation arguments to the early stopper. We define that the early stopper should evaluate every 5
# epochs with a patience of 20 epochs on the validation set. In order to continue training, we expect the model to
# obtain an improvement > 0.2% in Hits@10.
directory = "doctests/ablation/ex05_stopper"
ablation_pipeline(
    directory=directory,
    models=["ComplEx"],
    datasets=["Nations"],
    losses=["BCEAfterSigmoidLoss", "MarginRankingLoss"],
    training_loops=["LCWA"],
    optimizers=["Adam"],
    stopper="early",
    stopper_kwargs={
        "frequency": 5,
        "patience": 20,
        "relative_delta": 0.002,
        "metric": "hits@10",
    },
    # Fast testing configuration, make bigger in prod
    epochs=1,
    n_trials=1,
)

# %%
# After defining the ablation study, we need to define the HPO settings for each experiment within our ablation
# study. In PyKEEN, we use `Optuna <https://github.com/optuna/optuna>`_ as HPO framework. We set the number of HPO
# iterations for each experiment to 2 using the argument ``n_trials``, set a ``timeout`` of 300 seconds (the HPO
# will be terminated after ``n_trials`` or ``timeout`` seconds depending on what occurs first), the ``metric`` to
# optimize, define whether the metric should be maximized or minimized using the argument ``direction``, define
# random search as HPO algorithm using the argument ``sampler``, and finally define that we do not use a pruner
# for pruning unpromising trials (note that we use early stopping instead).
directory = "doctests/ablation/ex06_optuna_kwargs"
ablation_pipeline(
    directory=directory,
    models="ComplEx",
    datasets="Nations",
    losses=["BCEAfterSigmoidLoss", "MarginRankingLoss"],
    training_loops="LCWA",
    optimizers="Adam",
    # Fast testing configuration, make bigger in prod
    epochs=1,
    # Optuna-related arguments
    n_trials=2,
    timeout=300,
    metric="hits@10",
    direction="maximize",
    sampler="random",
    pruner="nop",
)

# %%
# To measure the variance in performance, we can additionally define how often we want to re-train and
# re-evaluate the best model of each ablation-experiment using the argument ``best_replicates``:
directory = "doctests/ablation/ex07_best_replicates"
ablation_pipeline(
    directory=directory,
    models=["ComplEx"],
    datasets=["Nations"],
    losses=["BCEAfterSigmoidLoss", "MarginRankingLoss"],
    training_loops=["LCWA"],
    optimizers=["Adam"],
    create_inverse_triples=[True, False],
    stopper="early",
    stopper_kwargs={
        "frequency": 5,
        "patience": 20,
        "relative_delta": 0.002,
        "metric": "hits@10",
    },
    # Fast testing configuration, make bigger in prod
    epochs=1,
    # Optuna-related arguments
    n_trials=2,
    timeout=300,
    metric="hits@10",
    direction="maximize",
    sampler="random",
    pruner="nop",
    best_replicates=5,
)

# %%
# Define Your Own HPO Ranges
# ---------------------------
# We provide default hyper-parameters/hyper-parameter ranges for each hyper-parameter. However, these default
# values/ranges do not ensure good performance. For the definition of hyper-parameter values/ranges, two
# dictionaries are essential, ``kwargs`` that is used to assign the hyper-parameters fixed values, and
# ``kwargs_ranges`` to define ranges of values from which to sample from.
#
# Let's start with assigning HPO ranges to hyper-parameters belonging to the interaction model, using the
# dictionary ``model_to_model_kwargs_ranges``. Because the ``scale`` is ``power_two``, the lower bound (``low``)
# equals to 4, and the upper bound ``high`` to 6, so the embedding dimension is sampled from the set {16, 32, 64}.
#
# Next, we fix the number of training epochs to 50 using the argument ``model_to_training_loop_to_training_kwargs``
# and define a range for the batch size using ``model_to_training_loop_to_training_kwargs_ranges``. We use these
# two dictionaries because the defined hyper-parameters are hyper-parameters of the training function (that is a
# function of the ``training_loop``).
#
# Finally, we define a range for the learning rate which is a hyper-parameter of the optimizer. We decided to use
# Adam as an optimizer, and defined a ``log`` scale for the learning rate, i.e., the learning rate is sampled from
# the interval [0.001, 0.1).

# Define HPO ranges
model_to_model_kwargs_ranges = {
    "ComplEx": {
        "embedding_dim": {
            "type": "int",
            "low": 4,
            "high": 6,
            "scale": "power_two",
        }
    }
}

model_to_training_loop_to_training_kwargs = {
    "ComplEx": {
        "lcwa": {
            "num_epochs": 50,
        }
    }
}

model_to_training_loop_to_training_kwargs_ranges = {
    "ComplEx": {
        "lcwa": {
            "label_smoothing": {
                "type": "float",
                "low": 0.001,
                "high": 1.0,
                "scale": "log",
            },
            "batch_size": {
                "type": "int",
                "low": 7,
                "high": 9,
                "scale": "power_two",
            },
        }
    }
}

model_to_optimizer_to_optimizer_kwargs_ranges = {
    "ComplEx": {
        "adam": {
            "lr": {
                "type": "float",
                "low": 0.001,
                "high": 0.1,
                "scale": "log",
            }
        }
    }
}

# %%
# Now that we defined our own hyper-parameter values/ranges, let's have a look at the overall configuration. We
# are expected to provide the arguments ``datasets``, ``models``, ``losses``, ``optimizers``, and
# ``training_loops`` to :func:`~pykeen.ablation.ablation_pipeline`. For all other components and hyper-parameters,
# PyKEEN provides default values/ranges. However, for achieving optimal performance, we should carefully define
# the hyper-parameter values/ranges ourselves, as shown above.
ablation_pipeline(
    metadata={"title": "Ablation Study Over Nations for ComplEx."},
    models=["ComplEx"],
    datasets=["Nations"],
    losses=["BCEAfterSigmoidLoss"],
    training_loops=["lcwa"],
    optimizers=["adam"],
    create_inverse_triples=[True, False],
    stopper="early",
    stopper_kwargs={
        "frequency": 5,
        "patience": 20,
        "relative_delta": 0.002,
        "metric": "hits@10",
    },
    model_to_model_kwargs_ranges=model_to_model_kwargs_ranges,
    model_to_training_loop_to_training_kwargs=model_to_training_loop_to_training_kwargs,
    model_to_optimizer_to_optimizer_kwargs_ranges=model_to_optimizer_to_optimizer_kwargs_ranges,
    directory="doctests/ablation/ex08_custom_ranges",
    best_replicates=5,
    n_trials=2,
    timeout=300,
    metric="hits@10",
    direction="maximize",
    sampler="random",
    pruner="nop",
)
