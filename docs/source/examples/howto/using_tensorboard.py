"""Using Tensorboard as a result tracker."""

# %%
from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="tensorboard",
)

# %%
pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="tensorboard",
    result_tracker_kwargs={
        "experiment_name": "rotate-kinships",
    },
)

# %%
pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="tensorboard",
    result_tracker_kwargs={
        "experiment_path": "tb-logs/rotate-kinships",
    },
)

# %%
from pykeen.hpo import hpo_pipeline

hpo_pipeline_result = hpo_pipeline(
    n_trials=30,
    dataset="Nations",
    model="TransE",
    result_tracker="tensorboard",
)
