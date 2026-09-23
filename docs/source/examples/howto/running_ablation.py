"""Running an ablation study."""

# %%
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
