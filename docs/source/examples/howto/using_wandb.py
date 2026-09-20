"""Using Weights and Biases as a result tracker."""

# %%
from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="wandb",
    result_tracker_kwargs={
        "project": "pykeen_project",
    },
)

# %%
pipeline_result = pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="wandb",
    result_tracker_kwargs={
        "project": "pykeen_project",
        "tags": "experiment-1",
    },
)

# %%
from pykeen.hpo import hpo_pipeline

hpo_pipeline_result = hpo_pipeline(
    model="RotatE",
    dataset="Kinships",
    result_tracker="wandb",
    result_tracker_kwargs={
        "project": "pykeen_project",
        "tags": "new run",
        "reinit": True,
    },
)
